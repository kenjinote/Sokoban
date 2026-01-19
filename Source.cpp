#pragma comment(linker,"\"/manifestdependency:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")
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
#include <set>
#include <thread>
#include <atomic>
#include <mutex>
#include <future>

#include "sqlite3.h"
#include "resource.h"

using namespace DirectX;

// --------------------------------------------------------------------------
// 定数・グローバル変数
// --------------------------------------------------------------------------
const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 600;
const char* DB_FILENAME = "sokoban_v6.db";

const int MAP_INFO_SIZE = 128;
const int SHADOW_MAP_SIZE = 1024;

// ★目標ステージ作成数
const int TARGET_STAGE_COUNT = 10;

// リストボックス用ID
#define IDC_STAGE_LIST 2001

const UINT MSAA_COUNT = 4;

enum TileType {
    TILE_NONE = -1, // ★追加: 虚空（描画されない、通れない）
    TILE_FLOOR = 0, TILE_WALL = 1, TILE_BOX = 2, TILE_GOAL = 3,
    TILE_BOX_ON_GOAL = 4, TILE_PLAYER_START = 8, TILE_PLAYER_ON_GOAL = 9
};

enum GameState { STATE_PLAY, STATE_FADE_OUT, STATE_FADE_IN, STATE_SELECT };

HWND g_hWnd = nullptr;
ID3D11Device* g_pd3dDevice = nullptr;
ID3D11DeviceContext* g_pImmediateContext = nullptr;
IDXGISwapChain* g_pSwapChain = nullptr;
ID3D11RenderTargetView* g_pRenderTargetView = nullptr;
ID3D11Texture2D* g_pDepthStencilBuffer = nullptr;
ID3D11DepthStencilView* g_pDepthStencilView = nullptr;

ID3D11VertexShader* g_pVertexShader = nullptr;
ID3D11PixelShader* g_pPixelShader = nullptr;
ID3D11VertexShader* g_pVSBake = nullptr;
ID3D11PixelShader* g_pPSBake = nullptr;

ID3D11InputLayout* g_pVertexLayout = nullptr;
ID3D11Buffer* g_pVertexBuffer = nullptr;
ID3D11Buffer* g_pConstantBuffer = nullptr;
ID3D11RasterizerState* g_pRasterState = nullptr;
ID3D11DepthStencilState* g_pDepthStencilState = nullptr;

ID3D11ShaderResourceView* g_pTextureRV = nullptr;
ID3D11ShaderResourceView* g_pCardboardTextureRV = nullptr;
ID3D11SamplerState* g_pSamplerLinear = nullptr;

ID3D11Texture2D* g_pMapTexture = nullptr;
ID3D11ShaderResourceView* g_pMapTextureRV = nullptr;
ID3D11SamplerState* g_pSamplerPoint = nullptr;

ID3D11Texture2D* g_pShadowMapTex = nullptr;
ID3D11RenderTargetView* g_pShadowMapRTV = nullptr;
ID3D11ShaderResourceView* g_pShadowMapSRV = nullptr;

D3D11_VIEWPORT g_Viewport = { 0 };
D3D11_VIEWPORT g_ShadowViewport = { 0 };

#define SAFE_RELEASE(p) { if(p) { (p)->Release(); (p)=nullptr; } }

// --------------------------------------------------------------------------
// シェーダー (省略なし)
// --------------------------------------------------------------------------
const char* g_shaderHlsl = R"(
Texture2D txDiffuse : register(t0); 
Texture2D txMapInfo : register(t1); 
Texture2D txShadowMap : register(t2); 

SamplerState samLinear : register(s0);
SamplerState samPoint  : register(s1); 

cbuffer ConstantBuffer : register(b0) {
    matrix World; matrix View; matrix Projection;
    float4 BaseColor; float4 LightPos; float4 LightParams;
    float4 CameraPos; float4 MaterialParams; float4 GameParams; 
    float4 MapSize; 
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

struct BAKE_PS_INPUT {
    float4 Pos : SV_POSITION;
    float2 Tex : TEXCOORD0;
};

BAKE_PS_INPUT VS_Bake(uint id : SV_VertexID) {
    BAKE_PS_INPUT output;
    output.Tex = float2((id << 1) & 2, id & 2);
    output.Pos = float4(output.Tex * float2(2, -2) + float2(-1, 1), 0, 1);
    return output;
}

bool IsOccluded(float3 pos) {
    if (pos.y < -0.01) return false;
    float2 uv = (float2(pos.x, pos.z) + 0.5f) / MapSize.z;
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) return true;
    float block = txMapInfo.SampleLevel(samPoint, uv, 0).r;
    if (block > 0.5 && pos.y < 1.05) return true;
    return false;
}

float3 GetRandomPointOnSphere(int index, int totalSamples, float randSeed) {
    float phi = 2.399963 * float(index) + randSeed * 62.8;
    float z = 1.0 - (2.0 * float(index) / float(totalSamples - 1)); 
    float r = sqrt(max(0.0, 1.0 - z * z));
    return float3(r * cos(phi), r * sin(phi), z);
}

float ComputeAreaShadow(float3 worldPos, float3 normal, int rayCount, float randSeed) {
    float3 lightCenter = LightPos.xyz;
    float lightRadius = 1.5f; 
    float visibleCount = 0.0;
    float3 biasOrigin = worldPos + normal * 0.01; 
    float distToLight = length(lightCenter - worldPos);
    if (distToLight > 25.0) return 1.0;
    const int STEPS = 64; 
    float stepSize = distToLight / float(STEPS);
    [loop]
    for(int i = 0; i < rayCount; ++i) {
        float3 rndDir = GetRandomPointOnSphere(i, rayCount, randSeed + float(i)*0.01);
        float3 targetPos = lightCenter + rndDir * lightRadius;
        float3 rayVec = targetPos - biasOrigin;
        float rayLen = length(rayVec);
        float3 rayDir = normalize(rayVec);
        float3 rayPos = biasOrigin;
        bool hit = false;
        for(int s = 0; s < STEPS; ++s) {
            rayPos += rayDir * stepSize;
            if (length(rayPos - biasOrigin) >= rayLen) break;
            if (IsOccluded(rayPos)) { hit = true; break; }
        }
        if (!hit) visibleCount += 1.0;
    }
    return visibleCount / float(rayCount);
}

float4 PS_Bake(BAKE_PS_INPUT input) : SV_Target {
    float3 worldPos = float3(input.Tex.x * MapSize.x - 0.5f, 0.0f, input.Tex.y * MapSize.y - 0.5f);
    if (length(worldPos - LightPos.xyz) > 22.0f) return float4(1.0, 1.0, 1.0, 1.0);
    if (IsOccluded(worldPos + float3(0, 0.05, 0))) { return float4(0.65, 0.65, 0.65, 1.0); }
    float noise = frac(sin(dot(input.Tex, float2(12.9898, 78.233))) * 43758.5453);
    float shadow = ComputeAreaShadow(worldPos, float3(0,1,0), 128, noise);
    float finalShadow = lerp(0.65, 1.0, shadow);
    return float4(finalShadow, finalShadow, finalShadow, 1.0);
}

float4 PS(PS_INPUT input) : SV_Target {
    float3 normal = normalize(input.Normal);
    float3 lightDir = normalize(LightPos.xyz - input.WorldPos);
    float3 viewDir = normalize(CameraPos.xyz - input.WorldPos);
    float dist = length(LightPos.xyz - input.WorldPos);
    float attenuation = saturate(1.0f - dist / (LightParams.x * 3.0)); 
    attenuation = pow(attenuation, 0.5f) * LightParams.y;
    float3 objColor = BaseColor.rgb;
    if (GameParams.y > 0.5f && GameParams.y < 1.5f) { 
        float3 texColor = txDiffuse.Sample(samLinear, input.Tex).rgb;
        objColor = texColor * BaseColor.rgb * 1.1f; 
    }
    float shadowFactor = 1.0f;
    if (GameParams.y < 0.5f) { 
        float2 shadowUV = (input.WorldPos.xz + 0.5f) / MapSize.xy;
        float2 texelSize = float2(1.0 / 1024.0, 1.0 / 1024.0);
        float sum = 0.0;
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                float2 offset = float2(float(x), float(y)) * texelSize * 1.5; 
                sum += txShadowMap.Sample(samLinear, saturate(shadowUV + offset)).r;
            }
        }
        shadowFactor = sum / 9.0;
    } 
    else if (GameParams.y > 1.5f) { shadowFactor = 1.0f; }
    else {
        if (normal.y > 0.9) { shadowFactor = 1.0f; } 
        else {
            if (dist > 20.0f) { shadowFactor = 1.0f; } else {
                float noise = frac(sin(dot(input.WorldPos.xz, float2(12.9898, 78.233))) * 43758.5453);
                float rawShadow = ComputeAreaShadow(input.WorldPos, normal, 32, noise);
                shadowFactor = lerp(0.65, 1.0, rawShadow);
            }
        }
    }
    float3 finalColor;
    if (GameParams.z > 0.5f) { 
        float pulse = (sin(GameParams.w * 3.0f) + 1.0f) * 0.5f; 
        float intensity = 0.8f + pulse * 0.4f; 
        float2 fromCenter = input.Tex - float2(0.5f, 0.5f);
        float glow = saturate(1.0f - length(fromCenter) * 1.8f);
        glow = pow(glow, 2.0f); 
        finalColor = objColor * intensity + float3(0.4f, 0.8f, 0.4f) * glow * 0.8f;
    }
    else {
        float3 ambient = float3(0.10f, 0.10f, 0.10f) * objColor;
        float diffFactor = max(0.0f, dot(normal, lightDir));
        float3 diffuse = diffFactor * attenuation * objColor;
        float3 halfVec = normalize(lightDir + viewDir);
        float specFactor = pow(max(0.0f, dot(normal, halfVec)), MaterialParams.z);
        float specular = float3(0.2f, 0.2f, 0.2f) * specFactor * MaterialParams.y * attenuation;
        finalColor = (ambient + diffuse + specular) * shadowFactor;
    }
    finalColor = pow(max(finalColor, 0.0f), 1.0/2.2); 
    float fade = 1.0f - saturate(GameParams.x);
    return float4(finalColor * fade, BaseColor.a);
}
)";

struct SimpleVertex { XMFLOAT3 Pos; XMFLOAT3 Normal; XMFLOAT2 Tex; };
struct ConstantBuffer {
    XMMATRIX mWorld; XMMATRIX mView; XMMATRIX mProjection;
    XMFLOAT4 vBaseColor; XMFLOAT4 vLightPos; XMFLOAT4 vLightParams;
    XMFLOAT4 vCameraPos; XMFLOAT4 vMaterialParams; XMFLOAT4 vGameParams;
    XMFLOAT4 vMapSize;
};

// --------------------------------------------------------------------------
// Helper Functions
// --------------------------------------------------------------------------
float frac(float v) { return v - floor(v); }
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
            if (isMortar) { float val = (0.6f + noise * 0.5f) * 255.0f; r = g = b = (uint32_t)min(255.0f, val); }
            else {
                float baseR = 0.8f; float baseG = 0.35f; float baseB = 0.15f;
                if ((brickX + brickY) % 3 == 0) { baseR *= 0.9f; baseG *= 0.9f; baseB *= 0.9f; }
                if ((brickX + brickY) % 5 == 0) { baseR *= 1.1f; baseG *= 1.05f; }
                r = (uint32_t)min(255.0f, (baseR + noise) * 255.0f); g = (uint32_t)min(255.0f, (baseG + noise) * 255.0f); b = (uint32_t)min(255.0f, (baseB + noise) * 255.0f);
            }
            textureData[y * width + x] = 0xFF000000 | (b << 16) | (g << 8) | r;
        }
    }
    D3D11_TEXTURE2D_DESC desc = {}; desc.Width = width; desc.Height = height; desc.MipLevels = 1; desc.ArraySize = 1; desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; desc.SampleDesc.Count = 1; desc.Usage = D3D11_USAGE_DEFAULT; desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initData = {}; initData.pSysMem = textureData.data(); initData.SysMemPitch = width * 4;
    ID3D11Texture2D* pTexture = nullptr; HRESULT hr = pDevice->CreateTexture2D(&desc, &initData, &pTexture);
    if (FAILED(hr)) return hr;
    hr = pDevice->CreateShaderResourceView(pTexture, nullptr, ppSRV); pTexture->Release(); return hr;
}
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
            bool isBrace1 = (abs(u - v) < braceSize * 0.7f); bool isBrace2 = (abs((u + v) - 1.0f) < braceSize * 0.7f);
            float heightMap = 0.0f; if (isFrame) heightMap = 1.0f; else if (isBrace1 || isBrace2) heightMap = 0.9f; else { heightMap = 0.5f; grain *= 0.8f; }
            float shadow = 1.0f; float highlight = 0.0f;
            if (!isFrame) {
                float du = min(abs(u - frameSize), abs(u - (1.0f - frameSize))); float dv = min(abs(v - frameSize), abs(v - (1.0f - frameSize)));
                if (min(du, dv) < 0.02f) shadow = 0.5f; if (isFrame && (u < 0.02f || v < 0.02f)) highlight = 0.2f;
            }
            if (!isFrame && !isBrace1 && !isBrace2) {
                float d1 = abs(u - v) - braceSize * 0.7f; float d2 = abs((u + v) - 1.0f) - braceSize * 0.7f;
                if ((d1 > 0 && d1 < 0.02f) || (d2 > 0 && d2 < 0.02f)) shadow = 0.4f;
            }
            float r = 0.70f; float g = 0.50f; float b = 0.30f;
            float light = heightMap * shadow * grain + highlight;
            float dirt = ((rand() % 100) / 100.0f) * 0.1f - 0.05f;
            r = r * light + dirt; g = g * light + dirt; b = b * light + dirt;
            uint32_t ir = (uint32_t)min(255.0f, max(0.0f, r * 255.0f)); uint32_t ig = (uint32_t)min(255.0f, max(0.0f, g * 255.0f)); uint32_t ib = (uint32_t)min(255.0f, max(0.0f, b * 255.0f));
            textureData[y * width + x] = 0xFF000000 | (ib << 16) | (ig << 8) | ir;
        }
    }
    D3D11_TEXTURE2D_DESC desc = {}; desc.Width = width; desc.Height = height; desc.MipLevels = 1; desc.ArraySize = 1; desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; desc.SampleDesc.Count = 1; desc.Usage = D3D11_USAGE_DEFAULT; desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA initData = {}; initData.pSysMem = textureData.data(); initData.SysMemPitch = width * 4;
    ID3D11Texture2D* pTexture = nullptr; HRESULT hr = pDevice->CreateTexture2D(&desc, &initData, &pTexture);
    if (FAILED(hr)) return hr;
    hr = pDevice->CreateShaderResourceView(pTexture, nullptr, ppSRV); pTexture->Release(); return hr;
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
        D3D11_TEXTURE2D_DESC desc = {}; desc.Width = width; desc.Height = height; desc.MipLevels = 1; desc.ArraySize = 1; desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; desc.SampleDesc.Count = 1; desc.Usage = D3D11_USAGE_DEFAULT; desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        D3D11_SUBRESOURCE_DATA initData = {}; initData.pSysMem = buffer.data(); initData.SysMemPitch = width * 4;
        ID3D11Texture2D* pTexture = nullptr; hr = pDevice->CreateTexture2D(&desc, &initData, &pTexture);
        if (SUCCEEDED(hr)) { hr = pDevice->CreateShaderResourceView(pTexture, nullptr, ppSRV); pTexture->Release(); }
    }
    if (pConverter) pConverter->Release(); if (pFrame) pFrame->Release(); if (pDecoder) pDecoder->Release(); if (pStream) pStream->Release(); if (pFactory) pFactory->Release(); return hr;
}

// --------------------------------------------------------------------------
// ソルバー・生成・DB管理
// --------------------------------------------------------------------------
struct Point { int x, y; bool operator==(const Point& o) const { return x == o.x && y == o.y; } };
struct CompressedState {
    std::vector<short> boxes; short playerReachIdx;
    bool operator<(const CompressedState& o) const { if (playerReachIdx != o.playerReachIdx) return playerReachIdx < o.playerReachIdx; return boxes < o.boxes; }
};

short GetNormalizedPlayerPos(int w, int h, const std::vector<bool>& walls, const std::vector<short>& boxPos, int currentPlayerIdx) {
    thread_local std::vector<bool> visited; thread_local std::vector<int> q; thread_local std::vector<bool> obstacles;
    if (visited.size() != w * h) { visited.resize(w * h); q.reserve(w * h); obstacles.resize(w * h); }
    std::fill(visited.begin(), visited.end(), false); q.clear(); std::copy(walls.begin(), walls.end(), obstacles.begin());
    for (short b : boxPos) obstacles[(b >> 8) * w + (b & 0xFF)] = true;
    if (obstacles[currentPlayerIdx]) return -1;
    q.push_back(currentPlayerIdx); visited[currentPlayerIdx] = true; int minIdx = currentPlayerIdx;
    int head = 0;
    while (head < q.size()) {
        int curr = q[head++]; if (curr < minIdx) minIdx = curr;
        int nextPos[4] = { curr - 1, curr + 1, curr - w, curr + w };
        for (int n : nextPos) if (n >= 0 && n < w * h && !obstacles[n] && !visited[n]) { visited[n] = true; q.push_back(n); }
    }
    return (short)minIdx;
}

// --------------------------------------------------------------------------
// 修正版1: SolveSokoban
// 探索上限を超えた場合、「高難易度」ではなく「解なし(-1)」として扱うように修正
// --------------------------------------------------------------------------
int SolveSokoban(int w, int h, const std::vector<int>& mapData, Point startPlayer, const std::vector<Point>& startBoxes, int limitDepth) {
    std::vector<bool> walls(w * h, false), goals(w * h, false);
    for (int i = 0; i < w * h; ++i) {
        if (mapData[i] == TILE_WALL || mapData[i] == TILE_NONE) walls[i] = true;
        if (mapData[i] == TILE_GOAL || mapData[i] == TILE_BOX_ON_GOAL) goals[i] = true;
    }
    std::vector<short> initialBoxes; for (const auto& b : startBoxes) initialBoxes.push_back((short)(b.y * 256 + b.x));
    std::sort(initialBoxes.begin(), initialBoxes.end());
    short normP = GetNormalizedPlayerPos(w, h, walls, initialBoxes, startPlayer.y * w + startPlayer.x);
    if (normP == -1) return -1;

    CompressedState startS; startS.boxes = initialBoxes; startS.playerReachIdx = normP;
    std::set<CompressedState> visited; std::vector<CompressedState> currentLevel, nextLevel;
    visited.insert(startS); currentLevel.push_back(startS);

    int moveX[4] = { -1, 1, 0, 0 }; int moveY[4] = { 0, 0, -1, 1 };

    // 探索ステート数の上限（これを超えたら計算打ち切り）
    const int MAX_DEPTH_SEARCH = 300000;
    int visitedStates = 0;

    thread_local std::vector<bool> reachable; thread_local std::vector<int> q; thread_local std::vector<int> boxMap;
    if (reachable.size() != w * h) { reachable.resize(w * h); q.reserve(w * h); boxMap.resize(w * h); }

    for (int depth = 0; depth < limitDepth; ++depth) {
        if (currentLevel.empty()) break;
        nextLevel.clear();

        for (const auto& s : currentLevel) {
            // ゴール判定
            bool allOK = true;
            for (short b : s.boxes) if (!goals[(b >> 8) * w + (b & 0xFF)]) { allOK = false; break; }
            if (allOK) return depth;

            std::fill(reachable.begin(), reachable.end(), false);
            std::fill(boxMap.begin(), boxMap.end(), -1);
            q.clear();
            for (int i = 0; i < s.boxes.size(); ++i) boxMap[(s.boxes[i] >> 8) * w + (s.boxes[i] & 0xFF)] = i;

            // プレイヤーの到達可能範囲から次の手を探索
            q.push_back(s.playerReachIdx); reachable[s.playerReachIdx] = true; int head = 0;
            while (head < q.size()) {
                int curr = q[head++]; int cx = curr % w; int cy = curr / w;
                for (int d = 0; d < 4; ++d) {
                    int nx = cx + moveX[d], ny = cy + moveY[d], nextPos = nx + ny * w;
                    if (nextPos < 0 || nextPos >= w * h || walls[nextPos]) continue;

                    int bIdx = boxMap[nextPos];
                    if (bIdx != -1) { // 箱がある
                        int pushDest = (nx + moveX[d]) + (ny + moveY[d]) * w;
                        if (pushDest >= 0 && pushDest < w * h && !walls[pushDest] && boxMap[pushDest] == -1) {
                            std::vector<short> nextBoxes = s.boxes;
                            nextBoxes[bIdx] = (short)((ny + moveY[d]) * 256 + (nx + moveX[d]));
                            std::sort(nextBoxes.begin(), nextBoxes.end());

                            short nextNormP = GetNormalizedPlayerPos(w, h, walls, nextBoxes, nextPos);
                            if (nextNormP != -1) {
                                CompressedState nextS; nextS.boxes = nextBoxes; nextS.playerReachIdx = nextNormP;
                                if (visited.find(nextS) == visited.end()) {
                                    visited.insert(nextS);
                                    nextLevel.push_back(nextS);
                                    visitedStates++;
                                }
                            }
                        }
                    }
                    else if (!reachable[nextPos]) { // 床
                        reachable[nextPos] = true; q.push_back(nextPos);
                    }
                }
            }
        }

        // ★修正点: 探索数が多すぎる場合は「解けない可能性が高い」として -1 を返す（以前は limitDepth を返していた）
        if (visitedStates > MAX_DEPTH_SEARCH) return -1;

        currentLevel = nextLevel;
    }
    return -1;
}

// --------------------------------------------------------------------------
// 修正版2: GenerateHighQualityStage
// 迷路生成ロジック + 厳格なクリア判定 + 不要な壁の削除
// --------------------------------------------------------------------------
void GenerateHighQualityStage(int stageId, int& outW, int& outH, std::vector<int>& outData) {
    const int W = 11; const int H = 11;

    // ステージIDに応じて目標手数を設定（20手〜最大150手）
    int targetMoves = 50 + (stageId * 2);
    if (targetMoves > 150) targetMoves = 150;

    std::atomic<bool> found(false);
    std::mutex resultMutex;
    std::vector<int> finalMapData;

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    std::vector<std::thread> workers;

    workers.clear();
    for (unsigned int i = 0; i < numThreads; ++i) {
        workers.emplace_back([&, i]() {
            // スレッドごとに異なる乱数シード
            std::mt19937 rng(stageId * 7777 + i * 999 + 12345);
            const int dList[4][2] = { {0,-1},{0,1},{-1,0},{1,0} };

            while (!found.load()) {
                // ■ 0. 生成タイプの決定 (30%で迷路モード)
                bool isMazeMode = (rng() % 10 < 3);

                // ■ 1. マップ生成 (Random Walk)
                std::vector<int> map(W * H, TILE_WALL);
                auto Set = [&](int x, int y, int t) { if (x > 0 && x < W - 1 && y>0 && y < H - 1) map[y * W + x] = t; };
                auto Get = [&](int x, int y) { if (x < 0 || x >= W || y < 0 || y >= H) return (int)TILE_WALL; return map[y * W + x]; };

                int cx = W / 2, cy = H / 2;
                Set(cx, cy, TILE_FLOOR);

                int floorCount = 1;
                int targetFloors = isMazeMode ? (int)((W - 2) * (H - 2) * 0.45) : (int)((W - 2) * (H - 2) * 0.55);

                int wx = cx, wy = cy;
                int maxSteps = 2000;

                for (int k = 0; k < maxSteps && floorCount < targetFloors; ++k) {
                    int d = rng() % 4;
                    wx += dList[d][0]; wy += dList[d][1];
                    if (wx < 1) wx = 1; if (wx > W - 2) wx = W - 2; if (wy < 1) wy = 1; if (wy > H - 2) wy = H - 2;

                    if (isMazeMode) {
                        // 1x1ブラシ (迷路風)
                        if (Get(wx, wy) == TILE_WALL) { Set(wx, wy, TILE_FLOOR); floorCount++; }
                    }
                    else {
                        // 2x2ブラシ (洞窟風)
                        for (int dy = 0; dy <= 1; ++dy) for (int dx = 0; dx <= 1; ++dx) {
                            int tx = wx + dx, ty = wy + dy;
                            if (Get(tx, ty) == TILE_WALL) { Set(tx, ty, TILE_FLOOR); floorCount++; }
                        }
                    }
                }

                // 迷路モードの場合は少しループを作る（詰み防止）
                if (isMazeMode) {
                    for (int k = 0; k < 4; ++k) {
                        int rx = 1 + rng() % (W - 2); int ry = 1 + rng() % (H - 2);
                        if (Get(rx, ry) == TILE_WALL) {
                            bool vLink = (Get(rx, ry - 1) == TILE_FLOOR && Get(rx, ry + 1) == TILE_FLOOR);
                            bool hLink = (Get(rx - 1, ry) == TILE_FLOOR && Get(rx + 1, ry) == TILE_FLOOR);
                            if (vLink || hLink) Set(rx, ry, TILE_FLOOR);
                        }
                    }
                }

                // ■ 2. ゴール領域の決定
                std::vector<Point> floors;
                for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) if (Get(x, y) == TILE_FLOOR) floors.push_back({ x,y });

                int boxCount = (isMazeMode ? 2 : 3) + (rng() % 3);
                if (floors.size() < (size_t)boxCount + 5) continue;

                // ゴールを固めて配置
                std::vector<Point> goals; std::vector<Point> candidates;
                Point seed = floors[rng() % floors.size()];
                candidates.push_back(seed); std::set<int> goalSet;

                while (goals.size() < (size_t)boxCount && !candidates.empty()) {
                    int idx = rng() % candidates.size(); Point p = candidates[idx]; candidates.erase(candidates.begin() + idx);
                    int pIdx = p.y * W + p.x; if (goalSet.count(pIdx)) continue;
                    goalSet.insert(pIdx); goals.push_back(p);
                    for (int d = 0; d < 4; ++d) {
                        Point n = { p.x + dList[d][0], p.y + dList[d][1] };
                        if (Get(n.x, n.y) == TILE_FLOOR && goalSet.find(n.y * W + n.x) == goalSet.end()) candidates.push_back(n);
                    }
                }
                if (goals.size() < (size_t)boxCount) continue;

                // ■ 3. 逆再生 (Pulling)
                std::vector<Point> boxes = goals; Point player;
                while (true) {
                    player = floors[rng() % floors.size()];
                    bool hit = false; for (auto g : goals) if (g.x == player.x && g.y == player.y) hit = true;
                    if (!hit) break;
                }

                int pullSteps = 400 + (rng() % 200);
                for (int s = 0; s < pullSteps; ++s) {
                    int bIdx = rng() % boxes.size(); Point b = boxes[bIdx]; int d = rng() % 4;
                    Point pullDest = { b.x + dList[d][0], b.y + dList[d][1] };
                    Point playerStand = { pullDest.x + dList[d][0], pullDest.y + dList[d][1] }; // プレイヤーが引くために立つ位置

                    if (Get(pullDest.x, pullDest.y) == TILE_WALL) continue;
                    if (Get(playerStand.x, playerStand.y) == TILE_WALL) continue;

                    bool hit = false;
                    for (auto& ob : boxes) { if (ob == pullDest || ob == playerStand) hit = true; }
                    if (hit) continue;

                    // プレイヤーが引く位置(playerStand)まで移動できるか確認
                    thread_local std::vector<bool> vis; thread_local std::vector<Point> q;
                    if (vis.size() != W * H) { vis.resize(W * H); q.reserve(W * H); }
                    std::fill(vis.begin(), vis.end(), false); q.clear();

                    // 箱を障害物として扱う
                    std::vector<bool> obs(W * H, false);
                    for (auto& ob : boxes) obs[ob.y * W + ob.x] = true;
                    if (obs[player.y * W + player.x]) continue;

                    q.push_back(player); vis[player.y * W + player.x] = true;
                    bool reach = false; int hHead = 0;
                    while (hHead < (int)q.size()) {
                        Point c = q[hHead++];
                        if (c == playerStand) { reach = true; break; }
                        for (int k = 0; k < 4; ++k) {
                            Point n = { c.x + dList[k][0], c.y + dList[k][1] }; int nIdx = n.y * W + n.x;
                            if (Get(n.x, n.y) != TILE_WALL && !obs[nIdx] && !vis[nIdx]) { vis[nIdx] = true; q.push_back(n); }
                        }
                    }
                    if (reach) {
                        boxes[bIdx] = pullDest;
                        player = b; // 箱があった位置にプレイヤーが移動（引いた動作）
                    }
                }

                // ■ 4. 検証 & 壁のシェイプアップ
                std::vector<int> finalMap = map;
                for (auto g : goals) finalMap[g.y * W + g.x] = TILE_GOAL;

                // ★ SolveSokoban は「解けない」「タイムアウト」の場合 -1 を返すようになったため、
                // diff >= targetMoves の条件だけで「確実に解ける かつ 難しい」ステージのみが通過する
                int diff = SolveSokoban(W, H, finalMap, player, boxes, targetMoves * 4);

                if (diff >= targetMoves) {
                    // 内部要素に接していない壁を削除 (Skinning)
                    std::vector<int> optimizedMap = finalMap;
                    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
                        if (finalMap[y * W + x] == TILE_WALL) {
                            bool isSkin = false;
                            for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                                int tx = x + dx; int ty = y + dy;
                                if (tx >= 0 && tx < W && ty >= 0 && ty < H) {
                                    int t = finalMap[ty * W + tx];
                                    if (t == TILE_FLOOR || t == TILE_GOAL) isSkin = true;
                                }
                            }
                            if (!isSkin) optimizedMap[y * W + x] = TILE_NONE;
                        }
                    }
                    finalMap = optimizedMap;

                    std::lock_guard<std::mutex> lock(resultMutex);
                    if (!found.load()) {
                        found.store(true);
                        outW = W; outH = H;
                        finalMapData = finalMap;
                        for (auto b : boxes) {
                            int idx = b.y * W + b.x;
                            if (finalMapData[idx] == TILE_GOAL) finalMapData[idx] = TILE_BOX_ON_GOAL;
                            else finalMapData[idx] = TILE_BOX;
                        }
                        int pIdx = player.y * W + player.x;
                        if (finalMapData[pIdx] == TILE_GOAL) finalMapData[pIdx] = TILE_PLAYER_ON_GOAL;
                        else finalMapData[pIdx] = TILE_PLAYER_START;
                    }
                }
            }
            });
    }
    while (!found.load()) { MSG msg; while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessage(&msg); } Sleep(50); }
    for (auto& t : workers) if (t.joinable()) t.join();
    outData = finalMapData;
}

void EnsureDatabaseExists() {
    sqlite3* db; if (sqlite3_open(DB_FILENAME, &db) != SQLITE_OK) return;
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS Stage (stage_id INTEGER, x INTEGER, y INTEGER, type INTEGER);", 0, 0, 0);
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS Progress (stage_id INTEGER PRIMARY KEY, cleared INTEGER);", 0, 0, 0);
    sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS Config (key TEXT PRIMARY KEY, value INTEGER);", 0, 0, 0);

    // ★既存の最大ステージIDを取得
    int maxStage = 0;
    sqlite3_stmt* stmt;
    sqlite3_prepare_v2(db, "SELECT MAX(stage_id) FROM Stage;", -1, &stmt, 0);
    if (sqlite3_step(stmt) == SQLITE_ROW && sqlite3_column_type(stmt, 0) != SQLITE_NULL) {
        maxStage = sqlite3_column_int(stmt, 0);
    }
    sqlite3_finalize(stmt);

    // ★足りない分を1つずつ生成・保存
    if (maxStage < TARGET_STAGE_COUNT) {
        sqlite3_stmt* insertStmt;
        sqlite3_prepare_v2(db, "INSERT INTO Stage (stage_id, x, y, type) VALUES (?, ?, ?, ?);", -1, &insertStmt, 0);

        for (int i = maxStage + 1; i <= TARGET_STAGE_COUNT; ++i) {
            int w, h; std::vector<int> d;
            GenerateHighQualityStage(i, w, h, d); // 時間がかかる処理

            char debugBuf[128];
            sprintf_s(debugBuf, "Generated Stage %d / %d\n", i, TARGET_STAGE_COUNT);
            OutputDebugStringA(debugBuf);

            // ★1ステージごとにコミット
            sqlite3_exec(db, "BEGIN TRANSACTION;", 0, 0, 0);
            for (int yy = 0; yy < h; ++yy) {
                for (int xx = 0; xx < w; ++xx) {
                    sqlite3_bind_int(insertStmt, 1, i);
                    sqlite3_bind_int(insertStmt, 2, xx);
                    sqlite3_bind_int(insertStmt, 3, yy);
                    sqlite3_bind_int(insertStmt, 4, d[yy * w + xx]);
                    sqlite3_step(insertStmt);
                    sqlite3_reset(insertStmt);
                }
            }
            sqlite3_exec(db, "COMMIT;", 0, 0, 0);
        }
        sqlite3_finalize(insertStmt);
    }
    sqlite3_close(db);
}

// ★重要: 前方宣言とグローバル変数
class Game;
extern Game g_Game;
WNDPROC g_pOldListBoxProc = nullptr;
LRESULT CALLBACK ListBoxProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

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
    bool m_mapDirty = true;
    HWND m_hListBox = nullptr; HFONT m_hFont = nullptr;

    Game() : m_moveSpeed(0.2f) {
        m_logicPos = { 0, 0 }; m_drawPos = { 0.0f, 0.0f };
        EnsureDatabaseExists(); // ★ここで不足分の生成が行われる
        LoadConfig();
        LoadStage(m_currentStage);
        StartFade(STATE_FADE_IN);
    }
    ~Game() { SaveConfig(); }

    void SaveConfig() {
        sqlite3* db; if (sqlite3_open(DB_FILENAME, &db) != SQLITE_OK) return;
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO Config (key, value) VALUES ('LastStage', ?);", -1, &stmt, 0);
        sqlite3_bind_int(stmt, 1, m_currentStage);
        sqlite3_step(stmt); sqlite3_finalize(stmt); sqlite3_close(db);
    }

    void LoadConfig() {
        sqlite3* db; if (sqlite3_open(DB_FILENAME, &db) != SQLITE_OK) return;
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, "SELECT value FROM Config WHERE key = 'LastStage';", -1, &stmt, 0);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            m_currentStage = sqlite3_column_int(stmt, 0);
            if (m_currentStage < 1) m_currentStage = 1;
        }
        sqlite3_finalize(stmt); sqlite3_close(db);
    }

    void MarkCleared() {
        sqlite3* db; if (sqlite3_open(DB_FILENAME, &db) != SQLITE_OK) return;
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, "INSERT OR REPLACE INTO Progress (stage_id, cleared) VALUES (?, 1);", -1, &stmt, 0);
        sqlite3_bind_int(stmt, 1, m_currentStage);
        sqlite3_step(stmt); sqlite3_finalize(stmt); sqlite3_close(db);
    }

    void UpdateTitle() {
        wchar_t title[256];
        if (m_isClear) swprintf_s(title, L"Sokoban - Stage #%d (CLEARED!) - Press 'Space' to Select Stage", m_currentStage);
        else swprintf_s(title, L"Sokoban - Stage #%d - Press 'Space' to Select Stage", m_currentStage);
        SetWindowText(g_hWnd, title);
    }

    void Reset() { StartFade(STATE_FADE_OUT); }
    void StartFade(GameState newState) { m_state = newState; m_animStartTime = GetTickCount64(); }

    void LoadStage(int stageId) {
        m_currentStage = stageId;
        sqlite3* db; if (sqlite3_open(DB_FILENAME, &db) != SQLITE_OK) return;
        sqlite3_stmt* stmt;
        const char* sqlSize = "SELECT MAX(x), MAX(y) FROM Stage WHERE stage_id = ?;";
        sqlite3_prepare_v2(db, sqlSize, -1, &stmt, 0); sqlite3_bind_int(stmt, 1, stageId);

        bool exists = false;
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            if (sqlite3_column_type(stmt, 0) != SQLITE_NULL) {
                m_mapWidth = sqlite3_column_int(stmt, 0) + 1;
                m_mapHeight = sqlite3_column_int(stmt, 1) + 1;
                exists = true;
            }
        }
        sqlite3_finalize(stmt);

        // ★修正: データが存在しない場合（最終ステージクリア後など）はステージ1へ
        if (!exists) {
            sqlite3_close(db);
            if (stageId > 1) LoadStage(1);
            return;
        }

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
        m_mapDirty = true;
        UpdateTitle();
    }

    void TryMove(int dx, int dy) {
        if (m_isClear || m_state != STATE_PLAY) return;
        int nextX = m_logicPos.x + dx; int nextY = m_logicPos.y + dy;
        if (nextX < 0 || nextX >= m_mapWidth || nextY < 0 || nextY >= m_mapHeight) return;
        int targetTile = m_mapData[nextY][nextX];

        // ★修正: 壁だけでなく NONE も移動不可
        if (targetTile == TILE_WALL || targetTile == TILE_NONE) return;

        if (targetTile == TILE_BOX || targetTile == TILE_BOX_ON_GOAL) {
            int beyondX = nextX + dx; int beyondY = nextY + dy;
            if (beyondX < 0 || beyondX >= m_mapWidth || beyondY < 0 || beyondY >= m_mapHeight) return;
            int beyondTile = m_mapData[beyondY][beyondX];
            if (beyondTile == TILE_FLOOR || beyondTile == TILE_GOAL) {
                if (beyondTile == TILE_FLOOR) m_mapData[beyondY][beyondX] = TILE_BOX; else m_mapData[beyondY][beyondX] = TILE_BOX_ON_GOAL;
                if (targetTile == TILE_BOX) m_mapData[nextY][nextX] = TILE_FLOOR; else m_mapData[nextY][nextX] = TILE_GOAL;
                m_logicPos = { nextX, nextY }; CheckClear(); m_mapDirty = true;
            }
        }
        else { m_logicPos = { nextX, nextY }; }
    }

    void CheckClear() {
        bool boxFound = false; for (const auto& row : m_mapData) for (int t : row) if (t == TILE_BOX) boxFound = true;
        if (!boxFound) { m_isClear = true; MarkCleared(); UpdateTitle(); }
    }

    void OpenStageSelect() {
        if (m_state == STATE_SELECT) return; m_state = STATE_SELECT;
        sqlite3* db; if (sqlite3_open(DB_FILENAME, &db) != SQLITE_OK) return;
        int maxStage = 0;
        sqlite3_stmt* stmt;
        sqlite3_prepare_v2(db, "SELECT MAX(stage_id) FROM Stage;", -1, &stmt, 0);
        if (sqlite3_step(stmt) == SQLITE_ROW) maxStage = sqlite3_column_int(stmt, 0);
        sqlite3_finalize(stmt);
        std::set<int> clearedStages;
        sqlite3_prepare_v2(db, "SELECT stage_id FROM Progress WHERE cleared = 1;", -1, &stmt, 0);
        while (sqlite3_step(stmt) == SQLITE_ROW) clearedStages.insert(sqlite3_column_int(stmt, 0));
        sqlite3_finalize(stmt); sqlite3_close(db);

        RECT rc; GetClientRect(g_hWnd, &rc);
        int w = rc.right / 2; int h = rc.bottom / 2; int x = rc.right / 4; int y = rc.bottom / 4;

        // ★不透明リストボックスとして作成 (WS_CHILD)
        m_hListBox = CreateWindowEx(WS_EX_CLIENTEDGE, L"LISTBOX", NULL,
            WS_CHILD | WS_VISIBLE | WS_VSCROLL | WS_BORDER | LBS_NOTIFY | LBS_HASSTRINGS,
            x, y, w, h, g_hWnd, (HMENU)IDC_STAGE_LIST, (HINSTANCE)GetWindowLongPtr(g_hWnd, GWLP_HINSTANCE), NULL);

        g_pOldListBoxProc = (WNDPROC)SetWindowLongPtr(m_hListBox, GWLP_WNDPROC, (LONG_PTR)ListBoxProc);
        HDC hdc = GetDC(g_hWnd);
        int nHeight = -MulDiv(12, GetDeviceCaps(hdc, LOGPIXELSY), 72); int nNewHeight = (int)(nHeight * 1.5f);
        m_hFont = CreateFont(nNewHeight, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Arial");
        ReleaseDC(g_hWnd, hdc);
        SendMessage(m_hListBox, WM_SETFONT, (WPARAM)m_hFont, TRUE);
        for (int i = 1; i <= maxStage; ++i) {
            wchar_t itemText[128];
            if (clearedStages.count(i)) swprintf_s(itemText, L"[O] Stage %d (Cleared)", i); else swprintf_s(itemText, L"[ ] Stage %d", i);
            int idx = (int)SendMessage(m_hListBox, LB_ADDSTRING, 0, (LPARAM)itemText);
            SendMessage(m_hListBox, LB_SETITEMDATA, idx, (LPARAM)i);
            if (i == m_currentStage) SendMessage(m_hListBox, LB_SETCURSEL, idx, 0);
        }
        ShowWindow(m_hListBox, SW_SHOW); UpdateWindow(m_hListBox); SetFocus(m_hListBox);
    }

    void CloseStageSelect(bool applySelection) {
        if (applySelection && m_hListBox) {
            int idx = (int)SendMessage(m_hListBox, LB_GETCURSEL, 0, 0);
            if (idx != LB_ERR) {
                int stageId = (int)SendMessage(m_hListBox, LB_GETITEMDATA, idx, 0);
                if (stageId > 0) { m_currentStage = stageId; LoadStage(m_currentStage); }
            }
        }
        if (m_hListBox) { DestroyWindow(m_hListBox); m_hListBox = nullptr; }
        if (m_hFont) { DeleteObject(m_hFont); m_hFont = nullptr; }
        m_state = STATE_PLAY; SetFocus(g_hWnd);
    }

    void Update() {
        if (m_state == STATE_SELECT) return;
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
        if (GetAsyncKeyState(VK_SPACE) & 0x8000) { OpenStageSelect(); return; }
        if (GetAsyncKeyState('R') & 0x8000) { Reset(); return; }
        if ((GetAsyncKeyState(VK_LEFT) & 0x8000) || (GetAsyncKeyState('A') & 0x8000)) dx = -1;
        else if ((GetAsyncKeyState(VK_RIGHT) & 0x8000) || (GetAsyncKeyState('D') & 0x8000)) dx = 1;
        else if ((GetAsyncKeyState(VK_UP) & 0x8000) || (GetAsyncKeyState('W') & 0x8000)) dy = 1;
        else if ((GetAsyncKeyState(VK_DOWN) & 0x8000) || (GetAsyncKeyState('S') & 0x8000)) dy = -1;

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

LRESULT CALLBACK ListBoxProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    if (uMsg == WM_KEYDOWN) {
        if (wParam == VK_ESCAPE || wParam == VK_SPACE) { g_Game.CloseStageSelect(false); return 0; }
    }
    return CallWindowProc(g_pOldListBoxProc, hWnd, uMsg, wParam, lParam);
}

// --------------------------------------------------------------------------
// Direct3D (省略なし)
// --------------------------------------------------------------------------
void ResizeD3D(UINT width, UINT height) {
    if (!g_pd3dDevice || !g_pSwapChain) return;
    g_pImmediateContext->OMSetRenderTargets(0, 0, 0);
    SAFE_RELEASE(g_pRenderTargetView); SAFE_RELEASE(g_pDepthStencilView); SAFE_RELEASE(g_pDepthStencilBuffer);
    g_pSwapChain->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, 0);
    ID3D11Texture2D* pBackBuffer = nullptr; g_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pBackBuffer);
    g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_pRenderTargetView);
    D3D11_TEXTURE2D_DESC bbDesc; pBackBuffer->GetDesc(&bbDesc); pBackBuffer->Release();
    D3D11_TEXTURE2D_DESC descDepth = {}; descDepth.Width = width; descDepth.Height = height; descDepth.MipLevels = 1; descDepth.ArraySize = 1;
    descDepth.Format = DXGI_FORMAT_D24_UNORM_S8_UINT; descDepth.SampleDesc.Count = bbDesc.SampleDesc.Count; descDepth.SampleDesc.Quality = bbDesc.SampleDesc.Quality;
    descDepth.Usage = D3D11_USAGE_DEFAULT; descDepth.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    g_pd3dDevice->CreateTexture2D(&descDepth, nullptr, &g_pDepthStencilBuffer);
    D3D11_DEPTH_STENCIL_VIEW_DESC descDSV = {}; descDSV.Format = descDepth.Format;
    if (bbDesc.SampleDesc.Count > 1) descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DMS; else descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    g_pd3dDevice->CreateDepthStencilView(g_pDepthStencilBuffer, &descDSV, &g_pDepthStencilView);
    g_pImmediateContext->OMSetRenderTargets(1, &g_pRenderTargetView, g_pDepthStencilView);
    float targetAspect = (float)SCREEN_WIDTH / SCREEN_HEIGHT; float windowAspect = (float)width / height;
    float viewW, viewH, viewX, viewY;
    if (windowAspect > targetAspect) { viewH = (float)height; viewW = viewH * targetAspect; viewX = (width - viewW) / 2.0f; viewY = 0; }
    else { viewW = (float)width; viewH = viewW / targetAspect; viewX = 0; viewY = (height - viewH) / 2.0f; }
    g_Viewport.Width = viewW; g_Viewport.Height = viewH; g_Viewport.TopLeftX = viewX; g_Viewport.TopLeftY = viewY; g_Viewport.MinDepth = 0.0f; g_Viewport.MaxDepth = 1.0f;
}

void UpdateMapTexture() {
    if (!g_Game.m_mapDirty || !g_pMapTexture) return;
    std::vector<uint8_t> mapData(MAP_INFO_SIZE * MAP_INFO_SIZE, 0);
    for (int y = 0; y < g_Game.m_mapHeight; ++y) for (int x = 0; x < g_Game.m_mapWidth; ++x) {
        int t = g_Game.m_mapData[y][x]; uint8_t val = 0;
        if (t == TILE_WALL || t == TILE_BOX || t == TILE_BOX_ON_GOAL) val = 255;
        if (x < MAP_INFO_SIZE && y < MAP_INFO_SIZE) mapData[y * MAP_INFO_SIZE + x] = val;
    }
    g_pImmediateContext->UpdateSubresource(g_pMapTexture, 0, nullptr, mapData.data(), MAP_INFO_SIZE, 0);
}

void BakeShadows() {
    if (!g_Game.m_mapDirty || !g_pShadowMapRTV) return;
    ID3D11RenderTargetView* prevRTV = nullptr; ID3D11DepthStencilView* prevDSV = nullptr; g_pImmediateContext->OMGetRenderTargets(1, &prevRTV, &prevDSV);
    D3D11_VIEWPORT prevVP; UINT numVP = 1; g_pImmediateContext->RSGetViewports(&numVP, &prevVP);
    g_pImmediateContext->OMSetRenderTargets(1, &g_pShadowMapRTV, nullptr); g_pImmediateContext->RSSetViewports(1, &g_ShadowViewport);
    float clearColor[4] = { 1.0f, 1.0f, 1.0f, 1.0f }; g_pImmediateContext->ClearRenderTargetView(g_pShadowMapRTV, clearColor);
    ConstantBuffer cb; cb.vMapSize = XMFLOAT4((float)g_Game.m_mapWidth, (float)g_Game.m_mapHeight, (float)MAP_INFO_SIZE, 0.0f);
    float centerX = (g_Game.m_mapWidth - 1) * 0.5f; float centerZ = (g_Game.m_mapHeight - 1) * 0.5f;
    cb.vLightPos = XMFLOAT4(centerX - 1.0f, 12.0f, centerZ - 1.0f, 1.0f);
    g_pImmediateContext->UpdateSubresource(g_pConstantBuffer, 0, nullptr, &cb, 0, 0);
    g_pImmediateContext->VSSetShader(g_pVSBake, nullptr, 0); g_pImmediateContext->PSSetShader(g_pPSBake, nullptr, 0);
    g_pImmediateContext->PSSetConstantBuffers(0, 1, &g_pConstantBuffer); g_pImmediateContext->PSSetShaderResources(1, 1, &g_pMapTextureRV);
    g_pImmediateContext->PSSetSamplers(1, 1, &g_pSamplerPoint);
    g_pImmediateContext->IASetVertexBuffers(0, 0, nullptr, nullptr, nullptr); g_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST); g_pImmediateContext->Draw(3, 0);
    g_pImmediateContext->OMSetRenderTargets(1, &prevRTV, prevDSV); g_pImmediateContext->RSSetViewports(1, &prevVP); SAFE_RELEASE(prevRTV); SAFE_RELEASE(prevDSV);
    g_Game.m_mapDirty = false;
}

HRESULT InitD3D(HWND hWnd) {
    HRESULT hr = S_OK;
    RECT rc; GetClientRect(hWnd, &rc); UINT width = rc.right - rc.left; UINT height = rc.bottom - rc.top;
    UINT createDeviceFlags = 0;
#ifdef _DEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0 }; D3D_FEATURE_LEVEL featureLevel;
    DXGI_SWAP_CHAIN_DESC sd = { 0 }; sd.BufferCount = 1; sd.BufferDesc.Width = width; sd.BufferDesc.Height = height; sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; sd.BufferDesc.RefreshRate.Numerator = 60; sd.BufferDesc.RefreshRate.Denominator = 1; sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; sd.OutputWindow = hWnd; sd.SampleDesc.Count = MSAA_COUNT; sd.SampleDesc.Quality = 0; sd.Windowed = TRUE;
    hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags, featureLevels, 1, D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pImmediateContext);
    if (FAILED(hr)) return hr;
    ResizeD3D(width, height);
    ID3DBlob* pVSBlob = nullptr; ID3DBlob* pPSBlob = nullptr; ID3DBlob* pVSBakeBlob = nullptr; ID3DBlob* pPSBakeBlob = nullptr;
    D3DCompile(g_shaderHlsl, strlen(g_shaderHlsl), nullptr, nullptr, nullptr, "VS", "vs_4_0", 0, 0, &pVSBlob, nullptr); g_pd3dDevice->CreateVertexShader(pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), nullptr, &g_pVertexShader);
    D3DCompile(g_shaderHlsl, strlen(g_shaderHlsl), nullptr, nullptr, nullptr, "PS", "ps_4_0", 0, 0, &pPSBlob, nullptr); g_pd3dDevice->CreatePixelShader(pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), nullptr, &g_pPixelShader);
    D3DCompile(g_shaderHlsl, strlen(g_shaderHlsl), nullptr, nullptr, nullptr, "VS_Bake", "vs_4_0", 0, 0, &pVSBakeBlob, nullptr); g_pd3dDevice->CreateVertexShader(pVSBakeBlob->GetBufferPointer(), pVSBakeBlob->GetBufferSize(), nullptr, &g_pVSBake);
    D3DCompile(g_shaderHlsl, strlen(g_shaderHlsl), nullptr, nullptr, nullptr, "PS_Bake", "ps_4_0", 0, 0, &pPSBakeBlob, nullptr); g_pd3dDevice->CreatePixelShader(pPSBakeBlob->GetBufferPointer(), pPSBakeBlob->GetBufferSize(), nullptr, &g_pPSBake);
    pPSBlob->Release(); pVSBakeBlob->Release(); pPSBakeBlob->Release();
    D3D11_INPUT_ELEMENT_DESC layout[] = { { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 }, { "NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 }, { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 }, };
    g_pd3dDevice->CreateInputLayout(layout, 3, pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), &g_pVertexLayout); pVSBlob->Release(); g_pImmediateContext->IASetInputLayout(g_pVertexLayout);
    SimpleVertex vertices[] = {
        { XMFLOAT3(-0.5f, 0.5f, -0.5f), XMFLOAT3(0,1,0), XMFLOAT2(0,0) }, { XMFLOAT3(-0.5f, 0.5f, 0.5f), XMFLOAT3(0,1,0), XMFLOAT2(0,1) }, { XMFLOAT3(0.5f, 0.5f, -0.5f), XMFLOAT3(0,1,0), XMFLOAT2(1,0) }, { XMFLOAT3(0.5f, 0.5f, -0.5f), XMFLOAT3(0,1,0), XMFLOAT2(1,0) }, { XMFLOAT3(-0.5f, 0.5f, 0.5f), XMFLOAT3(0,1,0), XMFLOAT2(0,1) }, { XMFLOAT3(0.5f, 0.5f, 0.5f), XMFLOAT3(0,1,0), XMFLOAT2(1,1) },
        { XMFLOAT3(-0.5f, -0.5f, -0.5f), XMFLOAT3(0,-1,0), XMFLOAT2(0,0) }, { XMFLOAT3(0.5f, -0.5f, -0.5f), XMFLOAT3(0,-1,0), XMFLOAT2(1,0) }, { XMFLOAT3(-0.5f, -0.5f, 0.5f), XMFLOAT3(0,-1,0), XMFLOAT2(0,1) }, { XMFLOAT3(-0.5f, -0.5f, 0.5f), XMFLOAT3(0,-1,0), XMFLOAT2(0,1) }, { XMFLOAT3(0.5f, -0.5f, -0.5f), XMFLOAT3(0,-1,0), XMFLOAT2(1,0) }, { XMFLOAT3(0.5f, -0.5f, 0.5f), XMFLOAT3(0,-1,0), XMFLOAT2(1,1) },
        { XMFLOAT3(-0.5f, -0.5f, -0.5f), XMFLOAT3(0,0,-1), XMFLOAT2(0,1) }, { XMFLOAT3(-0.5f, 0.5f, -0.5f), XMFLOAT3(0,0,-1), XMFLOAT2(0,0) }, { XMFLOAT3(0.5f, -0.5f, -0.5f), XMFLOAT3(0,0,-1), XMFLOAT2(1,1) }, { XMFLOAT3(0.5f, -0.5f, -0.5f), XMFLOAT3(0,0,-1), XMFLOAT2(1,1) }, { XMFLOAT3(-0.5f, 0.5f, -0.5f), XMFLOAT3(0,0,-1), XMFLOAT2(0,0) }, { XMFLOAT3(0.5f, 0.5f, -0.5f), XMFLOAT3(0,0,-1), XMFLOAT2(1,0) },
        { XMFLOAT3(-0.5f, -0.5f, 0.5f), XMFLOAT3(0,0,1), XMFLOAT2(1,1) }, { XMFLOAT3(0.5f, -0.5f, 0.5f), XMFLOAT3(0,0,1), XMFLOAT2(0,1) }, { XMFLOAT3(-0.5f, 0.5f, 0.5f), XMFLOAT3(0,0,1), XMFLOAT2(1,0) }, { XMFLOAT3(-0.5f, 0.5f, 0.5f), XMFLOAT3(0,0,1), XMFLOAT2(1,0) }, { XMFLOAT3(0.5f, -0.5f, 0.5f), XMFLOAT3(0,0,1), XMFLOAT2(0,1) }, { XMFLOAT3(0.5f, 0.5f, 0.5f), XMFLOAT3(0,0,1), XMFLOAT2(0,0) },
        { XMFLOAT3(-0.5f, -0.5f, 0.5f), XMFLOAT3(-1,0,0), XMFLOAT2(0,1) }, { XMFLOAT3(-0.5f, -0.5f, -0.5f), XMFLOAT3(-1,0,0), XMFLOAT2(1,1) }, { XMFLOAT3(-0.5f, 0.5f, 0.5f), XMFLOAT3(-1,0,0), XMFLOAT2(0,0) }, { XMFLOAT3(-0.5f, 0.5f, 0.5f), XMFLOAT3(-1,0,0), XMFLOAT2(0,0) }, { XMFLOAT3(-0.5f, -0.5f, -0.5f), XMFLOAT3(-1,0,0), XMFLOAT2(1,1) }, { XMFLOAT3(-0.5f, 0.5f, -0.5f), XMFLOAT3(-1,0,0), XMFLOAT2(1,0) },
        { XMFLOAT3(0.5f, -0.5f, -0.5f), XMFLOAT3(1,0,0), XMFLOAT2(0,1) }, { XMFLOAT3(0.5f, -0.5f, 0.5f), XMFLOAT3(1,0,0), XMFLOAT2(1,1) }, { XMFLOAT3(0.5f, 0.5f, -0.5f), XMFLOAT3(1,0,0), XMFLOAT2(0,0) }, { XMFLOAT3(0.5f, 0.5f, -0.5f), XMFLOAT3(1,0,0), XMFLOAT2(0,0) }, { XMFLOAT3(0.5f, -0.5f, 0.5f), XMFLOAT3(1,0,0), XMFLOAT2(1,1) }, { XMFLOAT3(0.5f, 0.5f, 0.5f), XMFLOAT3(1,0,0), XMFLOAT2(1,0) },
    };
    D3D11_BUFFER_DESC bd = { 0 }; bd.Usage = D3D11_USAGE_DEFAULT; bd.ByteWidth = sizeof(SimpleVertex) * 36; bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA InitData = { 0 }; InitData.pSysMem = vertices;
    g_pd3dDevice->CreateBuffer(&bd, &InitData, &g_pVertexBuffer);
    UINT stride = sizeof(SimpleVertex); UINT offset = 0; g_pImmediateContext->IASetVertexBuffers(0, 1, &g_pVertexBuffer, &stride, &offset);
    g_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    bd.ByteWidth = sizeof(ConstantBuffer); bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER; g_pd3dDevice->CreateBuffer(&bd, nullptr, &g_pConstantBuffer);
    D3D11_RASTERIZER_DESC rasterDesc = { 0 }; rasterDesc.FillMode = D3D11_FILL_SOLID; rasterDesc.CullMode = D3D11_CULL_NONE; rasterDesc.MultisampleEnable = TRUE; rasterDesc.AntialiasedLineEnable = TRUE;
    g_pd3dDevice->CreateRasterizerState(&rasterDesc, &g_pRasterState); g_pImmediateContext->RSSetState(g_pRasterState);
    CoInitialize(nullptr);
    hr = LoadTextureFromResource(g_pd3dDevice, L"WALL_JPG", L"JPG", &g_pTextureRV); if (FAILED(hr)) CreateProceduralBrickTexture(g_pd3dDevice, &g_pTextureRV);
    hr = LoadTextureFromResource(g_pd3dDevice, L"CARDBOARD_JPG", L"JPG", &g_pCardboardTextureRV); if (FAILED(hr)) CreateProceduralCardboardTexture(g_pd3dDevice, &g_pCardboardTextureRV);
    D3D11_SAMPLER_DESC sampDesc = {}; sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR; sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP; sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP; sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    g_pd3dDevice->CreateSamplerState(&sampDesc, &g_pSamplerLinear);
    D3D11_TEXTURE2D_DESC mapDesc = {}; mapDesc.Width = MAP_INFO_SIZE; mapDesc.Height = MAP_INFO_SIZE; mapDesc.MipLevels = 1; mapDesc.ArraySize = 1; mapDesc.Format = DXGI_FORMAT_R8_UNORM; mapDesc.SampleDesc.Count = 1; mapDesc.Usage = D3D11_USAGE_DEFAULT; mapDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    g_pd3dDevice->CreateTexture2D(&mapDesc, nullptr, &g_pMapTexture); g_pd3dDevice->CreateShaderResourceView(g_pMapTexture, nullptr, &g_pMapTextureRV);
    sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT; g_pd3dDevice->CreateSamplerState(&sampDesc, &g_pSamplerPoint);
    D3D11_TEXTURE2D_DESC shadowDesc = {}; shadowDesc.Width = SHADOW_MAP_SIZE; shadowDesc.Height = SHADOW_MAP_SIZE; shadowDesc.MipLevels = 1; shadowDesc.ArraySize = 1; shadowDesc.Format = DXGI_FORMAT_R8_UNORM; shadowDesc.SampleDesc.Count = 1; shadowDesc.Usage = D3D11_USAGE_DEFAULT; shadowDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    g_pd3dDevice->CreateTexture2D(&shadowDesc, nullptr, &g_pShadowMapTex); g_pd3dDevice->CreateRenderTargetView(g_pShadowMapTex, nullptr, &g_pShadowMapRTV); g_pd3dDevice->CreateShaderResourceView(g_pShadowMapTex, nullptr, &g_pShadowMapSRV);
    g_ShadowViewport.Width = (float)SHADOW_MAP_SIZE; g_ShadowViewport.Height = (float)SHADOW_MAP_SIZE; g_ShadowViewport.MinDepth = 0.0f; g_ShadowViewport.MaxDepth = 1.0f;
    return S_OK;
}
void CleanupD3D() {
    // 終了時にステージ保存
    g_Game.SaveConfig();

    if (g_pImmediateContext) g_pImmediateContext->ClearState();
    SAFE_RELEASE(g_pShadowMapSRV); SAFE_RELEASE(g_pShadowMapRTV); SAFE_RELEASE(g_pShadowMapTex);
    SAFE_RELEASE(g_pSamplerPoint); SAFE_RELEASE(g_pMapTextureRV); SAFE_RELEASE(g_pMapTexture);
    SAFE_RELEASE(g_pTextureRV); SAFE_RELEASE(g_pCardboardTextureRV);
    SAFE_RELEASE(g_pSamplerLinear); SAFE_RELEASE(g_pRasterState); SAFE_RELEASE(g_pDepthStencilState); SAFE_RELEASE(g_pConstantBuffer); SAFE_RELEASE(g_pVertexBuffer); SAFE_RELEASE(g_pVertexLayout);
    SAFE_RELEASE(g_pPixelShader); SAFE_RELEASE(g_pVertexShader); SAFE_RELEASE(g_pVSBake); SAFE_RELEASE(g_pPSBake);
    SAFE_RELEASE(g_pDepthStencilView); SAFE_RELEASE(g_pDepthStencilBuffer); SAFE_RELEASE(g_pRenderTargetView); SAFE_RELEASE(g_pSwapChain); SAFE_RELEASE(g_pImmediateContext); SAFE_RELEASE(g_pd3dDevice);
    CoUninitialize();
}

void Render() {
    g_Game.Update();
    if (g_Game.m_state == STATE_SELECT) return;

    UpdateMapTexture(); BakeShadows();

    // ★追加: 有効なタイル(TILE_NONE以外)が存在する範囲(バウンディングボックス)を計算
    int minX = g_Game.m_mapWidth;
    int maxX = 0;
    int minY = g_Game.m_mapHeight;
    int maxY = 0;
    bool foundContent = false;

    for (int y = 0; y < g_Game.m_mapHeight; ++y) {
        for (int x = 0; x < g_Game.m_mapWidth; ++x) {
            if (g_Game.m_mapData[y][x] != TILE_NONE) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
                foundContent = true;
            }
        }
    }

    // 万が一何もなかった場合のフォールバック
    if (!foundContent) { minX = 0; maxX = g_Game.m_mapWidth - 1; minY = 0; maxY = g_Game.m_mapHeight - 1; }

    float contentWidth = (float)(maxX - minX + 1);
    float contentHeight = (float)(maxY - minY + 1);
    float centerX = minX + (contentWidth - 1.0f) * 0.5f;
    float centerZ = minY + (contentHeight - 1.0f) * 0.5f;

    float ClearColor[4] = { 0.95f, 0.95f, 0.95f, 1.0f };
    g_pImmediateContext->ClearRenderTargetView(g_pRenderTargetView, ClearColor); g_pImmediateContext->ClearDepthStencilView(g_pDepthStencilView, D3D11_CLEAR_DEPTH, 1.0f, 0); g_pImmediateContext->RSSetViewports(1, &g_Viewport);

    float fovRad = XMConvertToRadians(45.0f); float aspectRatio = (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT;
    float tanHalfFov = tanf(fovRad * 0.5f);

    // ★修正: マップ全体サイズではなく、コンテンツサイズに合わせてカメラ距離を計算
    float distY = (contentHeight + 2.0f) * 0.5f / tanHalfFov;
    float distX = (contentWidth + 2.0f) * 0.5f / (tanHalfFov * aspectRatio);
    float camDist = max(max(distX, distY), 8.0f); // 最低距離を少し近づける

    // ★修正: 計算した centerX, centerZ を注視点にする
    XMVECTOR At = XMVectorSet(centerX, 0.0f, centerZ, 0.0f);
    XMVECTOR Eye = XMVectorSet(centerX, camDist, centerZ - (camDist * 0.2f), 0.0f);
    XMVECTOR Up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

    XMMATRIX mView = XMMatrixLookAtLH(Eye, At, Up); XMMATRIX mProjection = XMMatrixPerspectiveFovLH(fovRad, aspectRatio, 0.1f, 200.0f);
    ConstantBuffer cb; cb.mView = XMMatrixTranspose(mView); cb.mProjection = XMMatrixTranspose(mProjection);

    // ライト位置も中心に合わせる
    cb.vLightPos = XMFLOAT4(centerX, 15.0f, centerZ, 1.0f);
    cb.vLightParams = XMFLOAT4(25.0f, 1.0f, 0.0f, 0.0f);
    float time = (float)(GetTickCount64() % 10000) / 1000.0f; cb.vGameParams = XMFLOAT4(g_Game.m_fadeAmount, 0.0f, 0.0f, time); cb.vMapSize = XMFLOAT4((float)g_Game.m_mapWidth, (float)g_Game.m_mapHeight, (float)MAP_INFO_SIZE, 0.0f);
    XMStoreFloat4(&cb.vCameraPos, Eye);
    g_pImmediateContext->VSSetShader(g_pVertexShader, nullptr, 0); g_pImmediateContext->PSSetShader(g_pPixelShader, nullptr, 0);
    g_pImmediateContext->VSSetConstantBuffers(0, 1, &g_pConstantBuffer); g_pImmediateContext->PSSetConstantBuffers(0, 1, &g_pConstantBuffer);
    g_pImmediateContext->PSSetShaderResources(1, 1, &g_pMapTextureRV); g_pImmediateContext->PSSetShaderResources(2, 1, &g_pShadowMapSRV);
    ID3D11SamplerState* samplers[] = { g_pSamplerLinear, g_pSamplerPoint }; g_pImmediateContext->PSSetSamplers(0, 2, samplers);

    auto IsWallOnly = [&](int tx, int ty) -> bool { if (tx < 0 || tx >= g_Game.m_mapWidth || ty < 0 || ty >= g_Game.m_mapHeight) return true; return g_Game.m_mapData[ty][tx] == TILE_WALL; };
    for (int y = 0; y < g_Game.m_mapHeight; ++y) for (int x = 0; x < g_Game.m_mapWidth; ++x) {
        int tile = g_Game.m_mapData[y][x];

        // TILE_NONE なら何も描画しない
        if (tile == TILE_NONE) continue;

        XMMATRIX mWorld = XMMatrixScaling(1.0f, 0.1f, 1.0f) * XMMatrixTranslation((float)x, -0.55f, (float)y);
        cb.mWorld = XMMatrixTranspose(mWorld); cb.vBaseColor = XMFLOAT4(0.4f, 0.7f, 0.4f, 1.0f); cb.vMaterialParams = XMFLOAT4(0.3f, 0.0f, 1.0f, 0.0f); cb.vGameParams.y = 0.0f; cb.vGameParams.z = 0.0f;
        g_pImmediateContext->UpdateSubresource(g_pConstantBuffer, 0, nullptr, &cb, 0, 0); g_pImmediateContext->Draw(6, 0);

        if (tile == TILE_WALL) {
            g_pImmediateContext->PSSetShaderResources(0, 1, &g_pTextureRV); mWorld = XMMatrixScaling(1.0f, 1.2f, 1.0f) * XMMatrixTranslation((float)x, 0.1f, (float)y); cb.mWorld = XMMatrixTranspose(mWorld); cb.vBaseColor = XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f); cb.vMaterialParams = XMFLOAT4(0.5f, 0.5f, 32.0f, 0.0f); cb.vGameParams.y = 1.0f; cb.vGameParams.z = 0.0f;
            g_pImmediateContext->UpdateSubresource(g_pConstantBuffer, 0, nullptr, &cb, 0, 0); g_pImmediateContext->Draw(6, 0);
            if (!IsWallOnly(x, y - 1)) g_pImmediateContext->Draw(6, 12); if (!IsWallOnly(x, y + 1)) g_pImmediateContext->Draw(6, 18); if (!IsWallOnly(x - 1, y)) g_pImmediateContext->Draw(6, 24); if (!IsWallOnly(x + 1, y)) g_pImmediateContext->Draw(6, 30);
        }
        else if (tile == TILE_BOX || tile == TILE_BOX_ON_GOAL) {
            g_pImmediateContext->PSSetShaderResources(0, 1, &g_pCardboardTextureRV); mWorld = XMMatrixScaling(0.85f, 0.85f, 0.85f) * XMMatrixTranslation((float)x, 0.0f, (float)y); cb.mWorld = XMMatrixTranspose(mWorld);
            if (tile == TILE_BOX_ON_GOAL) cb.vBaseColor = XMFLOAT4(0.6f, 0.4f, 0.3f, 1.0f); else cb.vBaseColor = XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
            cb.vMaterialParams = XMFLOAT4(0.8f, 0.8f, 64.0f, 0.0f); cb.vGameParams.y = 1.0f; cb.vGameParams.z = 0.0f;
            g_pImmediateContext->UpdateSubresource(g_pConstantBuffer, 0, nullptr, &cb, 0, 0); g_pImmediateContext->Draw(36, 0);
        }
        else if (tile == TILE_GOAL) {
            mWorld = XMMatrixScaling(0.6f, 0.05f, 0.6f) * XMMatrixTranslation((float)x, -0.48f, (float)y); cb.mWorld = XMMatrixTranspose(mWorld); cb.vBaseColor = XMFLOAT4(0.6f, 1.0f, 0.6f, 1.0f); cb.vMaterialParams = XMFLOAT4(0.1f, 1.0f, 8.0f, 0.0f); cb.vGameParams.y = 0.0f; cb.vGameParams.z = 1.0f;
            g_pImmediateContext->UpdateSubresource(g_pConstantBuffer, 0, nullptr, &cb, 0, 0); g_pImmediateContext->Draw(6, 0);
        }
    }
    XMMATRIX mPlayerWorld = XMMatrixScaling(0.7f, 0.7f, 0.7f) * XMMatrixTranslation(g_Game.m_drawPos.x, 0.0f, g_Game.m_drawPos.y);
    cb.mWorld = XMMatrixTranspose(mPlayerWorld); cb.vBaseColor = XMFLOAT4(1.0f, 0.2f, 0.5f, 1.0f); cb.vMaterialParams = XMFLOAT4(0.0f, 1.0f, 64.0f, 0.0f); cb.vGameParams.y = 2.0f; cb.vGameParams.z = 0.0f;
    g_pImmediateContext->UpdateSubresource(g_pConstantBuffer, 0, nullptr, &cb, 0, 0); g_pImmediateContext->Draw(36, 0);
    g_pSwapChain->Present(1, 0);
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_LBUTTONDOWN:
        if (g_Game.m_state == STATE_SELECT) {
            g_Game.CloseStageSelect(false);
            return 0;
        }
        else if (g_Game.m_state == STATE_PLAY) {
            g_Game.OpenStageSelect();
            return 0;
        }
        break;
    case WM_COMMAND:
        if (LOWORD(wParam) == IDC_STAGE_LIST && HIWORD(wParam) == LBN_SELCHANGE) {
            g_Game.CloseStageSelect(true);
            return 0;
        }
        break;
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE) {
            if (g_Game.m_state == STATE_SELECT) g_Game.CloseStageSelect(false);
            else g_Game.Reset();
            return 0;
        }
        if (wParam == 'Q') { DestroyWindow(hWnd); return 0; }
        return 0;
    case WM_SIZE: ResizeD3D(LOWORD(lParam), HIWORD(lParam)); return 0;
    case WM_DESTROY: PostQuitMessage(0); return 0;
    } return DefWindowProc(hWnd, message, wParam, lParam);
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, LPWSTR, int nCmdShow) {
    WNDCLASSEX wcex = { sizeof(WNDCLASSEX) }; wcex.style = CS_HREDRAW | CS_VREDRAW; wcex.lpfnWndProc = WndProc; wcex.hInstance = hInstance;
    wcex.hCursor = LoadCursor(nullptr, IDC_ARROW); wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_ICON1));
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1); wcex.lpszClassName = L"SokobanComplete"; RegisterClassEx(&wcex);
    g_hWnd = CreateWindow(L"SokobanComplete", L"Sokoban", WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN, CW_USEDEFAULT, CW_USEDEFAULT, SCREEN_WIDTH, SCREEN_HEIGHT, nullptr, nullptr, hInstance, nullptr);

    // ★重要: 起動直後にタイトルを表示するために追加
    g_Game.UpdateTitle();

    if (!g_hWnd) return FALSE; ShowWindow(g_hWnd, nCmdShow);
    if (FAILED(InitD3D(g_hWnd))) { CleanupD3D(); return 0; }
    MSG msg = { 0 }; while (msg.message != WM_QUIT) { if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessage(&msg); } else { Render(); } }
    CleanupD3D(); return (int)msg.wParam;
}