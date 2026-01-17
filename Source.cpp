#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "windowscodecs.lib")

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <vector>
#include <string>
#include <iostream>
#include <random>
#include <algorithm>
#include <wincodec.h>

#include "sqlite3.h"
#include "resource.h"

using namespace DirectX;

// --------------------------------------------------------------------------
// 定数・グローバル変数
// --------------------------------------------------------------------------
// ★ 修正: 定数名を SCREEN_WIDTH / SCREEN_HEIGHT に戻しました
const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 600;
const char* DB_FILENAME = "sokoban_v4.db";
#define IDC_RESET_BUTTON 1001

enum TileType {
    TILE_FLOOR = 0, TILE_WALL = 1, TILE_BOX = 2, TILE_GOAL = 3,
    TILE_BOX_ON_GOAL = 4, TILE_PLAYER_START = 8, TILE_PLAYER_ON_GOAL = 9
};

enum GameState { STATE_PLAY, STATE_FADE_OUT, STATE_FADE_IN };

HWND g_hWnd = nullptr;
ID3D11Device* g_pd3dDevice = nullptr;
ID3D11DeviceContext* g_pImmediateContext = nullptr;
IDXGISwapChain* g_pSwapChain = nullptr;
ID3D11RenderTargetView* g_pRenderTargetView = nullptr;
ID3D11Texture2D* g_pDepthStencilBuffer = nullptr;
ID3D11DepthStencilView* g_pDepthStencilView = nullptr;
ID3D11VertexShader* g_pVertexShader = nullptr;
ID3D11PixelShader* g_pPixelShader = nullptr;
ID3D11InputLayout* g_pVertexLayout = nullptr;
ID3D11Buffer* g_pVertexBuffer = nullptr;
ID3D11Buffer* g_pConstantBuffer = nullptr;
ID3D11RasterizerState* g_pRasterState = nullptr;
ID3D11DepthStencilState* g_pDepthStencilState = nullptr;

ID3D11ShaderResourceView* g_pTextureRV = nullptr;
ID3D11ShaderResourceView* g_pCardboardTextureRV = nullptr;
ID3D11SamplerState* g_pSamplerLinear = nullptr;

D3D11_VIEWPORT g_Viewport = { 0 };

#define SAFE_RELEASE(p) { if(p) { (p)->Release(); (p)=nullptr; } }

// --------------------------------------------------------------------------
// シェーダー
// --------------------------------------------------------------------------
const char* g_shaderHlsl = R"(
Texture2D txDiffuse : register(t0);
SamplerState samLinear : register(s0);

cbuffer ConstantBuffer : register(b0) {
    matrix World; matrix View; matrix Projection;
    float4 BaseColor; float4 LightPos; float4 LightParams;
    float4 CameraPos; float4 MaterialParams; float4 GameParams; 
}

struct VS_INPUT { 
    float4 Pos : POSITION; 
    float3 Normal : NORMAL; 
    float2 Tex : TEXCOORD0;
};
struct PS_INPUT { 
    float4 Pos : SV_POSITION; 
    float3 Normal : NORMAL; 
    float3 WorldPos : TEXCOORD0; 
    float2 Tex : TEXCOORD1;
};

PS_INPUT VS(VS_INPUT input) {
    PS_INPUT output = (PS_INPUT)0;
    float4 worldPos = mul(input.Pos, World);
    output.WorldPos = worldPos.xyz;
    output.Pos = mul(worldPos, View);
    output.Pos = mul(output.Pos, Projection);
    output.Normal = normalize(mul(input.Normal, (float3x3)World));
    output.Tex = input.Tex;
    return output;
}

float4 PS(PS_INPUT input) : SV_Target {
    float3 normal = normalize(input.Normal);
    float3 lightDir = normalize(LightPos.xyz - input.WorldPos);
    
    float dist = length(LightPos.xyz - input.WorldPos);
    float attenuation = saturate(1.0f - dist / LightParams.x);
    attenuation = pow(attenuation, 0.8f) * LightParams.y;

    float3 objColor = BaseColor.rgb;
    if (GameParams.y > 0.5f) {
        float3 texColor = txDiffuse.Sample(samLinear, input.Tex).rgb;
        objColor = texColor * BaseColor.rgb * 1.1f; 
    }

    float3 finalColor;

    if (GameParams.z > 0.5f) { // 発光
        float pulse = (sin(GameParams.w * 3.0f) + 1.0f) * 0.5f; 
        float intensity = 0.8f + pulse * 0.4f; 
        float2 fromCenter = input.Tex - float2(0.5f, 0.5f);
        float distCenter = length(fromCenter);
        float glow = saturate(1.0f - distCenter * 1.8f);
        glow = pow(glow, 2.0f); 
        finalColor = objColor * intensity + float3(0.4f, 0.8f, 0.4f) * glow * 0.8f;
    }
    else {
        float3 ambient = float3(0.25f, 0.25f, 0.25f) * objColor;
        float diffuse = max(0.0f, dot(normal, lightDir)) * attenuation * objColor;
        float3 viewDir = normalize(CameraPos.xyz - input.WorldPos);
        float3 halfVec = normalize(lightDir + viewDir);
        float spec = pow(max(0.0f, dot(normal, halfVec)), MaterialParams.z);
        float3 specular = float3(0.2f, 0.2f, 0.2f) * spec * MaterialParams.y * attenuation;
        finalColor = ambient + diffuse + specular;
    }

    // 彩度調整
    float luminance = dot(finalColor, float3(0.299f, 0.587f, 0.114f));
    float saturationFactor = 1.2f;
    finalColor = lerp(float3(luminance, luminance, luminance), finalColor, saturationFactor);

    float fade = 1.0f - saturate(GameParams.x);
    return float4(finalColor * fade, BaseColor.a);
}
)";

struct SimpleVertex {
    XMFLOAT3 Pos; XMFLOAT3 Normal; XMFLOAT2 Tex;
};

struct ConstantBuffer {
    XMMATRIX mWorld; XMMATRIX mView; XMMATRIX mProjection;
    XMFLOAT4 vBaseColor; XMFLOAT4 vLightPos; XMFLOAT4 vLightParams;
    XMFLOAT4 vCameraPos; XMFLOAT4 vMaterialParams; XMFLOAT4 vGameParams;
};

// --------------------------------------------------------------------------
// テクスチャ読み込み・生成ヘルパー
// --------------------------------------------------------------------------
float frac(float v) { return v - floor(v); }

// レンガテクスチャ生成
HRESULT CreateProceduralBrickTexture(ID3D11Device* pDevice, ID3D11ShaderResourceView** ppSRV) {
    const int width = 256; const int height = 256;
    std::vector<uint32_t> textureData(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float u = (float)x / width; float v = (float)y / height;
            int brickX = (int)(u * 4.0f + (int)(v * 8.0f) % 2 * 0.5f); int brickY = (int)(v * 8.0f);
            float edgeU = frac(u * 4.0f + (int)(v * 8.0f) % 2 * 0.5f); float edgeV = frac(v * 8.0f);
            bool isMortar = (edgeU < 0.05f || edgeV < 0.1f);
            float noise = ((rand() % 100) / 100.0f) * 0.2f;
            uint32_t r, g, b;
            if (isMortar) {
                float val = (0.6f + noise * 0.5f) * 255.0f; r = g = b = (uint32_t)min(255.0f, val);
            }
            else {
                float baseR = 0.8f; float baseG = 0.35f; float baseB = 0.15f;
                if ((brickX + brickY) % 3 == 0) { baseR *= 0.9f; baseG *= 0.9f; baseB *= 0.9f; }
                if ((brickX + brickY) % 5 == 0) { baseR *= 1.1f; baseG *= 1.05f; }
                r = (uint32_t)min(255.0f, (baseR + noise) * 255.0f);
                g = (uint32_t)min(255.0f, (baseG + noise) * 255.0f);
                b = (uint32_t)min(255.0f, (baseB + noise) * 255.0f);
            }
            textureData[y * width + x] = 0xFF000000 | (b << 16) | (g << 8) | r;
        }
    }
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width; desc.Height = height;
    desc.MipLevels = 1; desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1; desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    D3D11_SUBRESOURCE_DATA initData = {};
    initData.pSysMem = textureData.data();
    initData.SysMemPitch = width * 4;

    ID3D11Texture2D* pTexture = nullptr;
    HRESULT hr = pDevice->CreateTexture2D(&desc, &initData, &pTexture);
    if (FAILED(hr)) return hr;

    hr = pDevice->CreateShaderResourceView(pTexture, nullptr, ppSRV);
    pTexture->Release();
    return hr;
}

// 木箱テクスチャ生成
HRESULT CreateProceduralCardboardTexture(ID3D11Device* pDevice, ID3D11ShaderResourceView** ppSRV) {
    const int width = 256; const int height = 256;
    std::vector<uint32_t> textureData(width * height);
    srand(123);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float u = (float)x / width; float v = (float)y / height;
            float noiseGrain = ((rand() % 100) / 100.0f);
            float grain = sin((u * 10.0f + v * 50.0f + noiseGrain * 0.1f) * 3.14f);
            grain = grain * 0.05f + 0.95f;
            float frameSize = 0.15f; float braceSize = 0.12f;
            bool isFrame = (u < frameSize || u > 1.0f - frameSize || v < frameSize || v > 1.0f - frameSize);
            bool isBrace1 = (abs(u - v) < braceSize * 0.7f);
            bool isBrace2 = (abs((u + v) - 1.0f) < braceSize * 0.7f);
            float heightMap = 0.0f;
            if (isFrame) heightMap = 1.0f; else if (isBrace1 || isBrace2) heightMap = 0.9f; else { heightMap = 0.5f; grain *= 0.8f; }
            float shadow = 1.0f; float highlight = 0.0f;
            if (!isFrame) {
                float du = min(abs(u - frameSize), abs(u - (1.0f - frameSize)));
                float dv = min(abs(v - frameSize), abs(v - (1.0f - frameSize)));
                if (min(du, dv) < 0.02f) shadow = 0.5f;
                if (isFrame && (u < 0.02f || v < 0.02f)) highlight = 0.2f;
            }
            if (!isFrame && !isBrace1 && !isBrace2) {
                float d1 = abs(u - v) - braceSize * 0.7f; float d2 = abs((u + v) - 1.0f) - braceSize * 0.7f;
                if ((d1 > 0 && d1 < 0.02f) || (d2 > 0 && d2 < 0.02f)) shadow = 0.4f;
            }
            float r = 0.70f; float g = 0.50f; float b = 0.30f;
            float light = heightMap * shadow * grain + highlight;
            float dirt = ((rand() % 100) / 100.0f) * 0.1f - 0.05f;
            r = r * light + dirt; g = g * light + dirt; b = b * light + dirt;
            uint32_t ir = (uint32_t)min(255.0f, max(0.0f, r * 255.0f));
            uint32_t ig = (uint32_t)min(255.0f, max(0.0f, g * 255.0f));
            uint32_t ib = (uint32_t)min(255.0f, max(0.0f, b * 255.0f));
            textureData[y * width + x] = 0xFF000000 | (ib << 16) | (ig << 8) | ir;
        }
    }
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width; desc.Height = height;
    desc.MipLevels = 1; desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1; desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    D3D11_SUBRESOURCE_DATA initData = {};
    initData.pSysMem = textureData.data();
    initData.SysMemPitch = width * 4;

    ID3D11Texture2D* pTexture = nullptr;
    HRESULT hr = pDevice->CreateTexture2D(&desc, &initData, &pTexture);
    if (FAILED(hr)) return hr;

    hr = pDevice->CreateShaderResourceView(pTexture, nullptr, ppSRV);
    pTexture->Release();
    return hr;
}

HRESULT LoadTextureFromResource(ID3D11Device* pDevice, const wchar_t* resourceName, const wchar_t* resourceType, ID3D11ShaderResourceView** ppSRV) {
    HRESULT hr = S_OK;
    HRSRC hRes = FindResource(nullptr, resourceName, resourceType); if (!hRes) return E_FAIL;
    HGLOBAL hMem = LoadResource(nullptr, hRes); void* pData = LockResource(hMem); DWORD size = SizeofResource(nullptr, hRes);
    IWICImagingFactory* pFactory = nullptr; hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pFactory)); if (FAILED(hr)) return hr;
    IWICStream* pStream = nullptr; hr = pFactory->CreateStream(&pStream); if (SUCCEEDED(hr)) hr = pStream->InitializeFromMemory((BYTE*)pData, size);
    IWICBitmapDecoder* pDecoder = nullptr; if (SUCCEEDED(hr)) hr = pFactory->CreateDecoderFromStream(pStream, nullptr, WICDecodeMetadataCacheOnDemand, &pDecoder);
    IWICBitmapFrameDecode* pFrame = nullptr; if (SUCCEEDED(hr)) hr = pDecoder->GetFrame(0, &pFrame);
    IWICFormatConverter* pConverter = nullptr; if (SUCCEEDED(hr)) hr = pFactory->CreateFormatConverter(&pConverter);
    if (SUCCEEDED(hr)) hr = pConverter->Initialize(pFrame, GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone, nullptr, 0.f, WICBitmapPaletteTypeMedianCut);
    if (SUCCEEDED(hr)) {
        UINT width, height; pConverter->GetSize(&width, &height); std::vector<BYTE> buffer(width * height * 4);
        pConverter->CopyPixels(nullptr, width * 4, width * height * 4, buffer.data());
        D3D11_TEXTURE2D_DESC desc = {};
        desc.Width = width; desc.Height = height;
        desc.MipLevels = 1; desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        desc.SampleDesc.Count = 1; desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

        D3D11_SUBRESOURCE_DATA initData = {};
        initData.pSysMem = buffer.data();
        initData.SysMemPitch = width * 4;

        ID3D11Texture2D* pTexture = nullptr;
        hr = pDevice->CreateTexture2D(&desc, &initData, &pTexture);
        if (SUCCEEDED(hr)) { hr = pDevice->CreateShaderResourceView(pTexture, nullptr, ppSRV); pTexture->Release(); }
    }
    if (pConverter) pConverter->Release(); if (pFrame) pFrame->Release(); if (pDecoder) pDecoder->Release(); if (pStream) pStream->Release(); if (pFactory) pFactory->Release(); return hr;
}

// --------------------------------------------------------------------------
// SQLite ヘルパー
// --------------------------------------------------------------------------
struct Point { int x, y; };
void GenerateProceduralStage(int stageId, int& outW, int& outH, std::vector<int>& outData) {
    int sizeBase = 7 + (stageId / 10); if (sizeBase > 20) sizeBase = 20;
    int boxCount = 2 + (stageId / 15); if (boxCount > 8) boxCount = 8;
    int steps = 50 + (stageId * 8);
    outW = sizeBase; outH = sizeBase; outData.assign(outW * outH, TILE_WALL);
    std::mt19937 rng(stageId * 12345);
    int cx = outW / 2; int cy = outH / 2;
    std::vector<Point> goals, boxes;
    Point playerPos = { cx, cy }; outData[cy * outW + cx] = TILE_FLOOR;
    int placedBoxes = 0;
    while (placedBoxes < boxCount) {
        int rx = (rng() % (sizeBase - 4)) + 2; int ry = (rng() % (sizeBase - 4)) + 2;
        int idx = ry * outW + rx;
        if (outData[idx] == TILE_WALL) { outData[idx] = TILE_BOX_ON_GOAL; boxes.push_back({ rx, ry }); goals.push_back({ rx, ry }); placedBoxes++; }
    }
    outData[cy * outW + cx] = TILE_PLAYER_START; playerPos = { cx, cy };
    for (int i = 0; i < steps; ++i) {
        int dirs[4][2] = { {0,1}, {0,-1}, {1,0}, {-1,0} }; int d = rng() % 4;
        int dx = dirs[d][0]; int dy = dirs[d][1];
        int px = playerPos.x + dx; int py = playerPos.y + dy;
        int bx = playerPos.x - dx; int by = playerPos.y - dy;
        if (px > 0 && px < outW - 1 && py > 0 && py < outH - 1) {
            if (outData[py * outW + px] == TILE_WALL) outData[py * outW + px] = TILE_FLOOR;
            if (rng() % 10 < 7) { playerPos = { px, py }; }
            else {
                bool boxFound = false; int boxIdx = -1;
                for (int k = 0; k < boxes.size(); ++k) { if (boxes[k].x == bx && boxes[k].y == by) { boxFound = true; boxIdx = k; break; } }
                if (boxFound) {
                    bool blocked = false; for (const auto& b : boxes) { if (b.x == px && b.y == py) blocked = true; }
                    if (!blocked) { boxes[boxIdx] = playerPos; playerPos = { px, py }; }
                }
                else { playerPos = { px, py }; }
            }
        }
    }
    for (int y = 0; y < outH; ++y) for (int x = 0; x < outW; ++x) { if (outData[y * outW + x] != TILE_WALL) outData[y * outW + x] = TILE_FLOOR; }
    for (const auto& g : goals) outData[g.y * outW + g.x] = TILE_GOAL;
    for (const auto& b : boxes) { int idx = b.y * outW + b.x; if (outData[idx] == TILE_GOAL) outData[idx] = TILE_BOX_ON_GOAL; else outData[idx] = TILE_BOX; }
    int pIdx = playerPos.y * outW + playerPos.x;
    if (outData[pIdx] == TILE_GOAL) outData[pIdx] = TILE_PLAYER_ON_GOAL; else outData[pIdx] = TILE_PLAYER_START;
}

void CreateStageData(sqlite3* db) {
    sqlite3_stmt* stmt; sqlite3_exec(db, "BEGIN TRANSACTION;", 0, 0, 0);
    sqlite3_prepare_v2(db, "INSERT INTO Stage (stage_id, x, y, type) VALUES (?, ?, ?, ?);", -1, &stmt, 0);
    const char* s1 = "  #####   \n  #   #   \n  #$  #   \n###   ##  \n#  $   #  \n# # . ##  \n#   . #   \n#####@#   \n    ###   ";
    int y = 0, x = 0; const char* p = s1;
    while (*p) {
        if (*p == '\n') { y++; x = 0; p++; continue; } int t = 0;
    switch (*p) { case '#':t = 1; break; case '$':t = 2; break; case '.':t = 3; break; case '*':t = 4; break; case '@':t = 8; break; case '+':t = 9; break; }
                          sqlite3_bind_int(stmt, 1, 1); sqlite3_bind_int(stmt, 2, x); sqlite3_bind_int(stmt, 3, y); sqlite3_bind_int(stmt, 4, t);
                          sqlite3_step(stmt); sqlite3_reset(stmt); x++; p++;
    }

    for (int i = 2; i <= 100; ++i) {
        int w, h; std::vector<int> d; GenerateProceduralStage(i, w, h, d);
        for (int yy = 0; yy < h; ++yy) for (int xx = 0; xx < w; ++xx) {
            sqlite3_bind_int(stmt, 1, i); sqlite3_bind_int(stmt, 2, xx); sqlite3_bind_int(stmt, 3, yy);
            sqlite3_bind_int(stmt, 4, d[yy * w + xx]); sqlite3_step(stmt); sqlite3_reset(stmt);
        }
    }
    sqlite3_finalize(stmt); sqlite3_exec(db, "COMMIT;", 0, 0, 0);
}
void EnsureDatabaseExists() {
    FILE* f; if (fopen_s(&f, DB_FILENAME, "r") == 0) { if (f)fclose(f); return; }
    sqlite3* db; if (sqlite3_open(DB_FILENAME, &db) != SQLITE_OK)return;
    sqlite3_exec(db, "CREATE TABLE Stage (stage_id INTEGER, x INTEGER, y INTEGER, type INTEGER);", 0, 0, 0);
    CreateStageData(db); sqlite3_close(db);
}

// --------------------------------------------------------------------------
// Gameクラス
// --------------------------------------------------------------------------
class Game {
public:
    XMINT2 m_logicPos; XMFLOAT2 m_drawPos; float m_moveSpeed;
    int m_mapWidth = 0; int m_mapHeight = 0;
    std::vector<std::vector<int>> m_mapData;
    int m_currentStage = 1; bool m_isClear = false;
    ULONGLONG m_nextMoveTime = 0; int m_lastDx = 0; int m_lastDy = 0;
    const int INITIAL_DELAY = 300; const int REPEAT_RATE = 100;
    GameState m_state = STATE_PLAY; float m_fadeAmount = 0.0f;
    ULONGLONG m_animStartTime = 0; const int FADE_DURATION = 500;

    Game() : m_moveSpeed(0.2f) {
        m_logicPos = { 0, 0 }; m_drawPos = { 0.0f, 0.0f };
        EnsureDatabaseExists(); LoadStage(m_currentStage); StartFade(STATE_FADE_IN);
    }
    void Reset() { StartFade(STATE_FADE_OUT); }
    void StartFade(GameState newState) { m_state = newState; m_animStartTime = GetTickCount64(); }
    void LoadStage(int stageId) {
        sqlite3* db; if (sqlite3_open(DB_FILENAME, &db) != SQLITE_OK) return;
        sqlite3_stmt* stmt;
        const char* sqlSize = "SELECT MAX(x), MAX(y) FROM Stage WHERE stage_id = ?;";
        sqlite3_prepare_v2(db, sqlSize, -1, &stmt, 0); sqlite3_bind_int(stmt, 1, stageId);
        if (sqlite3_step(stmt) == SQLITE_ROW) { m_mapWidth = sqlite3_column_int(stmt, 0) + 1; m_mapHeight = sqlite3_column_int(stmt, 1) + 1; }
        else { m_mapWidth = 0; sqlite3_finalize(stmt); sqlite3_close(db); if (stageId > 1) { m_currentStage = 1; LoadStage(1); } return; }
        sqlite3_finalize(stmt);
        m_mapData.assign(m_mapHeight, std::vector<int>(m_mapWidth, TILE_FLOOR));
        const char* sqlData = "SELECT x, y, type FROM Stage WHERE stage_id = ?;";
        sqlite3_prepare_v2(db, sqlData, -1, &stmt, 0); sqlite3_bind_int(stmt, 1, stageId);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            int x = sqlite3_column_int(stmt, 0); int y = sqlite3_column_int(stmt, 1); int type = sqlite3_column_int(stmt, 2);
            if (x < 0 || x >= m_mapWidth || y < 0 || y >= m_mapHeight) continue;
            if (type == TILE_PLAYER_START) { m_logicPos = { x, y }; m_mapData[y][x] = TILE_FLOOR; }
            else if (type == TILE_PLAYER_ON_GOAL) { m_logicPos = { x, y }; m_mapData[y][x] = TILE_GOAL; }
            else { m_mapData[y][x] = type; }
        }
        sqlite3_finalize(stmt); sqlite3_close(db);
        m_drawPos = { (float)m_logicPos.x, (float)m_logicPos.y };
        m_isClear = false; m_lastDx = 0; m_lastDy = 0; m_nextMoveTime = 0;
    }
    void TryMove(int dx, int dy) {
        if (m_isClear || m_state != STATE_PLAY) return;
        int nextX = m_logicPos.x + dx; int nextY = m_logicPos.y + dy;
        if (nextX < 0 || nextX >= m_mapWidth || nextY < 0 || nextY >= m_mapHeight) return;
        int targetTile = m_mapData[nextY][nextX];
        if (targetTile == TILE_WALL) return;
        if (targetTile == TILE_BOX || targetTile == TILE_BOX_ON_GOAL) {
            int beyondX = nextX + dx; int beyondY = nextY + dy;
            if (beyondX < 0 || beyondX >= m_mapWidth || beyondY < 0 || beyondY >= m_mapHeight) return;
            int beyondTile = m_mapData[beyondY][beyondX];
            if (beyondTile == TILE_FLOOR || beyondTile == TILE_GOAL) {
                if (beyondTile == TILE_FLOOR) m_mapData[beyondY][beyondX] = TILE_BOX; else m_mapData[beyondY][beyondX] = TILE_BOX_ON_GOAL;
                if (targetTile == TILE_BOX) m_mapData[nextY][nextX] = TILE_FLOOR; else m_mapData[nextY][nextX] = TILE_GOAL;
                m_logicPos = { nextX, nextY }; CheckClear();
            }
        }
        else { m_logicPos = { nextX, nextY }; }
    }
    void CheckClear() {
        bool boxFound = false; for (const auto& row : m_mapData) for (int t : row) if (t == TILE_BOX) boxFound = true;
        if (!boxFound) m_isClear = true;
    }
    void Update() {
        ULONGLONG t = GetTickCount64();
        if (m_state == STATE_FADE_OUT) {
            float p = (float)(t - m_animStartTime) / (float)FADE_DURATION;
            if (p >= 1.0f) { m_fadeAmount = 1.0f; if (m_isClear) m_currentStage++; LoadStage(m_currentStage); StartFade(STATE_FADE_IN); }
            else m_fadeAmount = p;
            return;
        }
        else if (m_state == STATE_FADE_IN) {
            float p = (float)(t - m_animStartTime) / (float)FADE_DURATION;
            if (p >= 1.0f) { m_fadeAmount = 0.0f; m_state = STATE_PLAY; }
            else m_fadeAmount = 1.0f - p;
            return;
        }
        else m_fadeAmount = 0.0f;
        if (m_isClear && m_state == STATE_PLAY) { StartFade(STATE_FADE_OUT); return; }
        int dx = 0, dy = 0;
        if (GetAsyncKeyState('R') & 0x8000) { Reset(); return; }
        if (GetAsyncKeyState(VK_LEFT) & 0x8000) dx = -1; else if (GetAsyncKeyState(VK_RIGHT) & 0x8000) dx = 1;
        else if (GetAsyncKeyState(VK_UP) & 0x8000) dy = 1; else if (GetAsyncKeyState(VK_DOWN) & 0x8000) dy = -1;
        if (dx != 0 || dy != 0) {
            if (dx != m_lastDx || dy != m_lastDy) { TryMove(dx, dy); m_nextMoveTime = t + INITIAL_DELAY; m_lastDx = dx; m_lastDy = dy; }
            else if (t >= m_nextMoveTime) { TryMove(dx, dy); m_nextMoveTime = t + REPEAT_RATE; }
        }
        else { m_lastDx = 0; m_lastDy = 0; }
        m_drawPos.x = Lerp(m_drawPos.x, (float)m_logicPos.x, m_moveSpeed);
        m_drawPos.y = Lerp(m_drawPos.y, (float)m_logicPos.y, m_moveSpeed);
    }
    float Lerp(float s, float e, float p) { return s + (e - s) * p; }
};
Game g_Game;

// --------------------------------------------------------------------------
// Direct3D (リサイズ対応)
// --------------------------------------------------------------------------
void ResizeD3D(UINT width, UINT height) {
    if (!g_pd3dDevice || !g_pSwapChain) return;
    g_pImmediateContext->OMSetRenderTargets(0, 0, 0);
    SAFE_RELEASE(g_pRenderTargetView); SAFE_RELEASE(g_pDepthStencilView); SAFE_RELEASE(g_pDepthStencilBuffer);
    g_pSwapChain->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, 0);
    ID3D11Texture2D* pBackBuffer = nullptr;
    g_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pBackBuffer);
    g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_pRenderTargetView); pBackBuffer->Release();
    D3D11_TEXTURE2D_DESC descDepth = {}; descDepth.Width = width; descDepth.Height = height; descDepth.MipLevels = 1; descDepth.ArraySize = 1;
    descDepth.Format = DXGI_FORMAT_D24_UNORM_S8_UINT; descDepth.SampleDesc.Count = 1; descDepth.Usage = D3D11_USAGE_DEFAULT; descDepth.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    g_pd3dDevice->CreateTexture2D(&descDepth, nullptr, &g_pDepthStencilBuffer);
    D3D11_DEPTH_STENCIL_VIEW_DESC descDSV = {}; descDSV.Format = descDepth.Format; descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    g_pd3dDevice->CreateDepthStencilView(g_pDepthStencilBuffer, &descDSV, &g_pDepthStencilView);
    g_pImmediateContext->OMSetRenderTargets(1, &g_pRenderTargetView, g_pDepthStencilView);

    float targetAspect = (float)SCREEN_WIDTH / SCREEN_HEIGHT;
    float windowAspect = (float)width / height;
    float viewW, viewH, viewX, viewY;
    if (windowAspect > targetAspect) { viewH = (float)height; viewW = viewH * targetAspect; viewX = (width - viewW) / 2.0f; viewY = 0; }
    else { viewW = (float)width; viewH = viewW / targetAspect; viewX = 0; viewY = (height - viewH) / 2.0f; }
    g_Viewport.Width = viewW; g_Viewport.Height = viewH; g_Viewport.TopLeftX = viewX; g_Viewport.TopLeftY = viewY;
    g_Viewport.MinDepth = 0.0f; g_Viewport.MaxDepth = 1.0f;
}

HRESULT InitD3D(HWND hWnd) {
    HRESULT hr = S_OK;
    RECT rc; GetClientRect(hWnd, &rc); UINT width = rc.right - rc.left; UINT height = rc.bottom - rc.top;
    UINT createDeviceFlags = 0;
#ifdef _DEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0 }; D3D_FEATURE_LEVEL featureLevel;
    DXGI_SWAP_CHAIN_DESC sd = { 0 }; sd.BufferCount = 1; sd.BufferDesc.Width = width; sd.BufferDesc.Height = height; sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; sd.BufferDesc.RefreshRate.Numerator = 60; sd.BufferDesc.RefreshRate.Denominator = 1; sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; sd.OutputWindow = hWnd; sd.SampleDesc.Count = 1; sd.Windowed = TRUE;
    hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags, featureLevels, 1, D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pImmediateContext);
    if (FAILED(hr)) return hr;

    ResizeD3D(width, height);

    // シェーダー作成
    ID3DBlob* pVSBlob = nullptr; ID3DBlob* pPSBlob = nullptr;
    D3DCompile(g_shaderHlsl, strlen(g_shaderHlsl), nullptr, nullptr, nullptr, "VS", "vs_4_0", 0, 0, &pVSBlob, nullptr);
    g_pd3dDevice->CreateVertexShader(pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), nullptr, &g_pVertexShader);
    D3DCompile(g_shaderHlsl, strlen(g_shaderHlsl), nullptr, nullptr, nullptr, "PS", "ps_4_0", 0, 0, &pPSBlob, nullptr);
    g_pd3dDevice->CreatePixelShader(pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), nullptr, &g_pPixelShader);
    pPSBlob->Release();

    D3D11_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    g_pd3dDevice->CreateInputLayout(layout, 3, pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), &g_pVertexLayout);
    pVSBlob->Release(); g_pImmediateContext->IASetInputLayout(g_pVertexLayout);

    SimpleVertex vertices[] = {
        // Top (Y+) - Green (Floor)
        { XMFLOAT3(-0.5f, 0.5f, -0.5f), XMFLOAT3(0,1,0),  XMFLOAT2(0,0) }, { XMFLOAT3(-0.5f, 0.5f, 0.5f), XMFLOAT3(0,1,0), XMFLOAT2(0,1) }, { XMFLOAT3(0.5f, 0.5f, -0.5f), XMFLOAT3(0,1,0), XMFLOAT2(1,0) },
        { XMFLOAT3(0.5f, 0.5f, -0.5f), XMFLOAT3(0,1,0),   XMFLOAT2(1,0) }, { XMFLOAT3(-0.5f, 0.5f, 0.5f), XMFLOAT3(0,1,0), XMFLOAT2(0,1) }, { XMFLOAT3(0.5f, 0.5f, 0.5f), XMFLOAT3(0,1,0), XMFLOAT2(1,1) },
        // Bottom (Y-)
        { XMFLOAT3(-0.5f, -0.5f, -0.5f), XMFLOAT3(0,-1,0), XMFLOAT2(0,0) }, { XMFLOAT3(0.5f, -0.5f, -0.5f), XMFLOAT3(0,-1,0), XMFLOAT2(1,0) }, { XMFLOAT3(-0.5f, -0.5f, 0.5f), XMFLOAT3(0,-1,0), XMFLOAT2(0,1) },
        { XMFLOAT3(-0.5f, -0.5f, 0.5f), XMFLOAT3(0,-1,0),  XMFLOAT2(0,1) }, { XMFLOAT3(0.5f, -0.5f, -0.5f), XMFLOAT3(0,-1,0), XMFLOAT2(1,0) }, { XMFLOAT3(0.5f, -0.5f, 0.5f), XMFLOAT3(0,-1,0), XMFLOAT2(1,1) },
        // Front (Z-) - Wall Face
        { XMFLOAT3(-0.5f, -0.5f, -0.5f), XMFLOAT3(0,0,-1), XMFLOAT2(0,1) }, { XMFLOAT3(-0.5f, 0.5f, -0.5f), XMFLOAT3(0,0,-1), XMFLOAT2(0,0) }, { XMFLOAT3(0.5f, -0.5f, -0.5f), XMFLOAT3(0,0,-1), XMFLOAT2(1,1) },
        { XMFLOAT3(0.5f, -0.5f, -0.5f), XMFLOAT3(0,0,-1),  XMFLOAT2(1,1) }, { XMFLOAT3(-0.5f, 0.5f, -0.5f), XMFLOAT3(0,0,-1), XMFLOAT2(0,0) }, { XMFLOAT3(0.5f, 0.5f, -0.5f), XMFLOAT3(0,0,-1), XMFLOAT2(1,0) },
        // Back (Z+) - Wall Face
        { XMFLOAT3(-0.5f, -0.5f, 0.5f), XMFLOAT3(0,0,1),   XMFLOAT2(1,1) }, { XMFLOAT3(0.5f, -0.5f, 0.5f), XMFLOAT3(0,0,1), XMFLOAT2(0,1) }, { XMFLOAT3(-0.5f, 0.5f, 0.5f), XMFLOAT3(0,0,1), XMFLOAT2(1,0) },
        { XMFLOAT3(-0.5f, 0.5f, 0.5f), XMFLOAT3(0,0,1),    XMFLOAT2(1,0) }, { XMFLOAT3(0.5f, -0.5f, 0.5f), XMFLOAT3(0,0,1), XMFLOAT2(0,1) }, { XMFLOAT3(0.5f, 0.5f, 0.5f), XMFLOAT3(0,0,1), XMFLOAT2(0,0) },
        // Left (X-) - Wall Face
        { XMFLOAT3(-0.5f, -0.5f, 0.5f), XMFLOAT3(-1,0,0),  XMFLOAT2(0,1) }, { XMFLOAT3(-0.5f, -0.5f, -0.5f), XMFLOAT3(-1,0,0), XMFLOAT2(1,1) }, { XMFLOAT3(-0.5f, 0.5f, 0.5f), XMFLOAT3(-1,0,0), XMFLOAT2(0,0) },
        { XMFLOAT3(-0.5f, 0.5f, 0.5f), XMFLOAT3(-1,0,0),   XMFLOAT2(0,0) }, { XMFLOAT3(-0.5f, -0.5f, -0.5f), XMFLOAT3(-1,0,0), XMFLOAT2(1,1) }, { XMFLOAT3(-0.5f, 0.5f, -0.5f), XMFLOAT3(-1,0,0), XMFLOAT2(1,0) },
        // Right (X+) - Wall Face
        { XMFLOAT3(0.5f, -0.5f, -0.5f), XMFLOAT3(1,0,0),   XMFLOAT2(0,1) }, { XMFLOAT3(0.5f, -0.5f, 0.5f), XMFLOAT3(1,0,0), XMFLOAT2(1,1) }, { XMFLOAT3(0.5f, 0.5f, -0.5f), XMFLOAT3(1,0,0), XMFLOAT2(0,0) },
        { XMFLOAT3(0.5f, 0.5f, -0.5f), XMFLOAT3(1,0,0),    XMFLOAT2(0,0) }, { XMFLOAT3(0.5f, -0.5f, 0.5f), XMFLOAT3(1,0,0), XMFLOAT2(1,1) }, { XMFLOAT3(0.5f, 0.5f, 0.5f), XMFLOAT3(1,0,0), XMFLOAT2(1,0) },
    };

    D3D11_BUFFER_DESC bd = { 0 }; bd.Usage = D3D11_USAGE_DEFAULT; bd.ByteWidth = sizeof(SimpleVertex) * 36; bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA InitData = { 0 }; InitData.pSysMem = vertices;
    g_pd3dDevice->CreateBuffer(&bd, &InitData, &g_pVertexBuffer);
    UINT stride = sizeof(SimpleVertex); UINT offset = 0; g_pImmediateContext->IASetVertexBuffers(0, 1, &g_pVertexBuffer, &stride, &offset);
    g_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    bd.ByteWidth = sizeof(ConstantBuffer); bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER; g_pd3dDevice->CreateBuffer(&bd, nullptr, &g_pConstantBuffer);

    D3D11_RASTERIZER_DESC rasterDesc = { 0 };
    rasterDesc.FillMode = D3D11_FILL_SOLID;
    rasterDesc.CullMode = D3D11_CULL_NONE;
    g_pd3dDevice->CreateRasterizerState(&rasterDesc, &g_pRasterState); g_pImmediateContext->RSSetState(g_pRasterState);

    CoInitialize(nullptr);
    hr = LoadTextureFromResource(g_pd3dDevice, L"WALL_JPG", L"JPG", &g_pTextureRV);
    if (FAILED(hr)) CreateProceduralBrickTexture(g_pd3dDevice, &g_pTextureRV);
    hr = LoadTextureFromResource(g_pd3dDevice, L"CARDBOARD_JPG", L"JPG", &g_pCardboardTextureRV);
    if (FAILED(hr)) CreateProceduralCardboardTexture(g_pd3dDevice, &g_pCardboardTextureRV);

    D3D11_SAMPLER_DESC sampDesc = {};
    sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
    sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
    sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    g_pd3dDevice->CreateSamplerState(&sampDesc, &g_pSamplerLinear);

    return S_OK;
}
void CleanupD3D() {
    if (g_pImmediateContext) g_pImmediateContext->ClearState();
    SAFE_RELEASE(g_pTextureRV); SAFE_RELEASE(g_pCardboardTextureRV);
    SAFE_RELEASE(g_pSamplerLinear); SAFE_RELEASE(g_pRasterState); SAFE_RELEASE(g_pDepthStencilState); SAFE_RELEASE(g_pConstantBuffer); SAFE_RELEASE(g_pVertexBuffer); SAFE_RELEASE(g_pVertexLayout); SAFE_RELEASE(g_pPixelShader); SAFE_RELEASE(g_pVertexShader); SAFE_RELEASE(g_pDepthStencilView); SAFE_RELEASE(g_pDepthStencilBuffer); SAFE_RELEASE(g_pRenderTargetView); SAFE_RELEASE(g_pSwapChain); SAFE_RELEASE(g_pImmediateContext); SAFE_RELEASE(g_pd3dDevice);
    CoUninitialize();
}

// --------------------------------------------------------------------------
// 描画ループ
// --------------------------------------------------------------------------
void Render() {
    g_Game.Update();
    float ClearColor[4] = { 0.95f, 0.95f, 0.95f, 1.0f };
    g_pImmediateContext->ClearRenderTargetView(g_pRenderTargetView, ClearColor);
    g_pImmediateContext->ClearDepthStencilView(g_pDepthStencilView, D3D11_CLEAR_DEPTH, 1.0f, 0);
    g_pImmediateContext->RSSetViewports(1, &g_Viewport);

    float centerX = (g_Game.m_mapWidth - 1) * 0.5f;
    float centerZ = (g_Game.m_mapHeight - 1) * 0.5f;

    XMVECTOR At = XMVectorSet(centerX, 0.0f, centerZ, 0.0f);
    XMVECTOR Eye = XMVectorSet(centerX, 12.0f, centerZ - 2.5f, 0.0f);
    XMVECTOR Up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
    XMMATRIX mView = XMMatrixLookAtLH(Eye, At, Up);
    float aspectRatio = (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT;
    XMMATRIX mProjection = XMMatrixPerspectiveFovLH(XMConvertToRadians(45.0f), aspectRatio, 0.1f, 100.0f);

    ConstantBuffer cb;
    cb.mView = XMMatrixTranspose(mView); cb.mProjection = XMMatrixTranspose(mProjection);
    cb.vLightPos = XMFLOAT4(centerX, 10.0f, centerZ, 1.0f);
    cb.vLightParams = XMFLOAT4(25.0f, 1.0f, 0.0f, 0.0f);

    float time = (float)(GetTickCount64() % 10000) / 1000.0f;
    cb.vGameParams = XMFLOAT4(g_Game.m_fadeAmount, 0.0f, 0.0f, time);
    XMStoreFloat4(&cb.vCameraPos, Eye);

    g_pImmediateContext->VSSetShader(g_pVertexShader, nullptr, 0);
    g_pImmediateContext->PSSetShader(g_pPixelShader, nullptr, 0);
    g_pImmediateContext->VSSetConstantBuffers(0, 1, &g_pConstantBuffer);
    g_pImmediateContext->PSSetConstantBuffers(0, 1, &g_pConstantBuffer);
    g_pImmediateContext->PSSetSamplers(0, 1, &g_pSamplerLinear);

    auto IsWall = [&](int tx, int ty) -> bool {
        if (tx < 0 || tx >= g_Game.m_mapWidth || ty < 0 || ty >= g_Game.m_mapHeight) return false;
        return g_Game.m_mapData[ty][tx] == TILE_WALL;
        };

    for (int y = 0; y < g_Game.m_mapHeight; ++y) {
        for (int x = 0; x < g_Game.m_mapWidth; ++x) {
            XMMATRIX mWorld = XMMatrixScaling(1.0f, 0.1f, 1.0f) * XMMatrixTranslation((float)x, -0.55f, (float)y);
            cb.mWorld = XMMatrixTranspose(mWorld);
            cb.vBaseColor = XMFLOAT4(0.4f, 0.7f, 0.4f, 1.0f);
            cb.vMaterialParams = XMFLOAT4(0.3f, 0.2f, 16.0f, 0.0f);
            cb.vGameParams.y = 0.0f; cb.vGameParams.z = 0.0f;
            g_pImmediateContext->UpdateSubresource(g_pConstantBuffer, 0, nullptr, &cb, 0, 0);
            g_pImmediateContext->Draw(6, 0);

            int tile = g_Game.m_mapData[y][x];
            if (tile == TILE_WALL) {
                g_pImmediateContext->PSSetShaderResources(0, 1, &g_pTextureRV);
                mWorld = XMMatrixScaling(1.0f, 1.2f, 1.0f) * XMMatrixTranslation((float)x, 0.1f, (float)y);
                cb.mWorld = XMMatrixTranspose(mWorld);
                cb.vBaseColor = XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
                cb.vMaterialParams = XMFLOAT4(0.5f, 0.5f, 32.0f, 0.0f);
                cb.vGameParams.y = 1.0f;
                g_pImmediateContext->UpdateSubresource(g_pConstantBuffer, 0, nullptr, &cb, 0, 0);
                g_pImmediateContext->Draw(6, 0);
                if (!IsWall(x, y - 1)) g_pImmediateContext->Draw(6, 12);
                if (!IsWall(x, y + 1)) g_pImmediateContext->Draw(6, 18);
                if (!IsWall(x - 1, y)) g_pImmediateContext->Draw(6, 24);
                if (!IsWall(x + 1, y)) g_pImmediateContext->Draw(6, 30);
            }
            else if (tile == TILE_BOX || tile == TILE_BOX_ON_GOAL) {
                g_pImmediateContext->PSSetShaderResources(0, 1, &g_pCardboardTextureRV);
                mWorld = XMMatrixScaling(0.85f, 0.85f, 0.85f) * XMMatrixTranslation((float)x, 0.0f, (float)y);
                cb.mWorld = XMMatrixTranspose(mWorld);
                if (tile == TILE_BOX_ON_GOAL) cb.vBaseColor = XMFLOAT4(0.6f, 0.4f, 0.3f, 1.0f);
                else cb.vBaseColor = XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
                cb.vMaterialParams = XMFLOAT4(0.8f, 0.8f, 64.0f, 0.0f);
                cb.vGameParams.y = 1.0f; cb.vGameParams.z = 0.0f;
                g_pImmediateContext->UpdateSubresource(g_pConstantBuffer, 0, nullptr, &cb, 0, 0);
                g_pImmediateContext->Draw(36, 0);
            }
            else if (tile == TILE_GOAL) {
                mWorld = XMMatrixScaling(0.6f, 0.05f, 0.6f) * XMMatrixTranslation((float)x, -0.48f, (float)y);
                cb.mWorld = XMMatrixTranspose(mWorld);
                cb.vBaseColor = XMFLOAT4(0.6f, 1.0f, 0.6f, 1.0f);
                cb.vMaterialParams = XMFLOAT4(0.1f, 1.0f, 8.0f, 0.0f);
                cb.vGameParams.y = 0.0f; cb.vGameParams.z = 1.0f;
                g_pImmediateContext->UpdateSubresource(g_pConstantBuffer, 0, nullptr, &cb, 0, 0);
                g_pImmediateContext->Draw(6, 0);
            }
        }
    }
    XMMATRIX mPlayerWorld = XMMatrixScaling(0.7f, 0.7f, 0.7f) * XMMatrixTranslation(g_Game.m_drawPos.x, 0.0f, g_Game.m_drawPos.y);
    cb.mWorld = XMMatrixTranspose(mPlayerWorld);
    cb.vBaseColor = XMFLOAT4(1.0f, 0.2f, 0.5f, 1.0f);
    cb.vMaterialParams = XMFLOAT4(0.0f, 1.0f, 64.0f, 0.0f);
    cb.vGameParams.y = 0.0f; cb.vGameParams.z = 0.0f;
    g_pImmediateContext->UpdateSubresource(g_pConstantBuffer, 0, nullptr, &cb, 0, 0);
    g_pImmediateContext->Draw(36, 0);
    g_pSwapChain->Present(1, 0);
}

// --------------------------------------------------------------------------
// Win32 メイン
// --------------------------------------------------------------------------
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE) { g_Game.Reset(); return 0; }
        if (wParam == 'Q') { DestroyWindow(hWnd); return 0; }
        return 0;
    case WM_SIZE: ResizeD3D(LOWORD(lParam), HIWORD(lParam)); return 0;
    case WM_DESTROY: PostQuitMessage(0); return 0;
    } return DefWindowProc(hWnd, message, wParam, lParam);
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, LPWSTR, int nCmdShow) {
    WNDCLASSEX wcex = { sizeof(WNDCLASSEX) }; wcex.style = CS_HREDRAW | CS_VREDRAW; wcex.lpfnWndProc = WndProc; wcex.hInstance = hInstance;
    wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_ICON1));
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1); wcex.lpszClassName = L"SokobanComplete"; RegisterClassEx(&wcex);
    g_hWnd = CreateWindow(L"SokobanComplete", L"Sokoban - ESC:Reset / Q:Quit", WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN, CW_USEDEFAULT, CW_USEDEFAULT, SCREEN_WIDTH, SCREEN_HEIGHT, nullptr, nullptr, hInstance, nullptr);
    if (!g_hWnd) return FALSE;
    ShowWindow(g_hWnd, nCmdShow);
    if (FAILED(InitD3D(g_hWnd))) { CleanupD3D(); return 0; }
    MSG msg = { 0 }; while (msg.message != WM_QUIT) { if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessage(&msg); } else { Render(); } }
    CleanupD3D(); return (int)msg.wParam;
}