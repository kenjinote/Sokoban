#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <vector>
#include <cmath>

// リンク設定: d3d11.lib, d3dcompiler.lib

using namespace DirectX;

// --------------------------------------------------------------------------
// 定数・グローバル変数
// --------------------------------------------------------------------------
const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 600;
const float GRID_SIZE = 1.0f;

enum TileType { TILE_FLOOR = 0, TILE_WALL = 1, TILE_BOX = 2 };

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

#define SAFE_RELEASE(p) { if(p) { (p)->Release(); (p)=nullptr; } }

// --------------------------------------------------------------------------
// シェーダー (明るさを微調整)
// --------------------------------------------------------------------------
const char* g_shaderHlsl = R"(
cbuffer ConstantBuffer : register(b0) {
    matrix World;
    matrix View;
    matrix Projection;
    float4 BaseColor;
    float4 LightPos;
    float4 LightParams;
    float4 CameraPos;
    float4 MaterialParams;
}

struct VS_INPUT {
    float4 Pos : POSITION;
    float3 Normal : NORMAL;
};

struct PS_INPUT {
    float4 Pos : SV_POSITION;
    float3 Normal : NORMAL;
    float3 WorldPos : TEXCOORD0;
};

float rand(float2 co){ return frac(sin(dot(co.xy ,float2(12.9898,78.233))) * 43758.5453); }

PS_INPUT VS(VS_INPUT input) {
    PS_INPUT output = (PS_INPUT)0;
    float4 worldPos = mul(input.Pos, World);
    output.WorldPos = worldPos.xyz;
    output.Pos = mul(worldPos, View);
    output.Pos = mul(output.Pos, Projection);
    output.Normal = normalize(mul(input.Normal, (float3x3)World));
    return output;
}

float4 PS(PS_INPUT input) : SV_Target {
    float3 normal = normalize(input.Normal);
    float3 lightVec = LightPos.xyz - input.WorldPos;
    float dist = length(lightVec);
    float3 lightDir = normalize(lightVec);

    float attenuation = saturate(1.0f - dist / LightParams.x);
    attenuation = pow(attenuation, 0.8f);
    attenuation *= LightParams.y; // 強度を適用

    float3 viewDir = normalize(CameraPos.xyz - input.WorldPos);
    
    float woodGrainScale = 6.0f;
    float noise = rand(input.WorldPos.xz * 0.5f) * 0.2f;
    float distGrain = length(input.WorldPos.xz + float2(noise, noise));
    float grain = (sin(distGrain * woodGrainScale) + 1.0f) * 0.5f;
    grain = pow(grain, 0.5f); 
    float3 woodColor = lerp(BaseColor.rgb * 0.6f, BaseColor.rgb, grain * MaterialParams.x + (1.0f - MaterialParams.x));

    // --- ライティング合成 (調整箇所) ---
    // 1. 環境光 (Ambient): 0.7 -> 0.6 へ少し下げる
    float3 ambient = float3(0.6f, 0.6f, 0.65f) * woodColor;

    // 2. 拡散反射 (Diffuse): 光の色を 1.0 -> 0.9 へ少し抑える
    float diffuseFactor = max(0.0f, dot(normal, lightDir));
    float3 diffuse = float3(0.9f, 0.88f, 0.8f) * diffuseFactor * attenuation * woodColor;

    // 3. 鏡面反射 (Specular)
    float3 halfVec = normalize(lightDir + viewDir);
    float specFactor = pow(max(0.0f, dot(normal, halfVec)), MaterialParams.z);
    float3 specular = float3(1.0f, 1.0f, 1.0f) * specFactor * MaterialParams.y * attenuation;

    float3 finalColor = ambient + diffuse + specular;
    
    // 最終的な色が1.0を超えないようにsaturateする（念のため）
    return float4(saturate(finalColor), BaseColor.a);
}
)";

struct SimpleVertex { XMFLOAT3 Pos; XMFLOAT3 Normal; };
struct ConstantBuffer {
    XMMATRIX mWorld; XMMATRIX mView; XMMATRIX mProjection;
    XMFLOAT4 vBaseColor;
    XMFLOAT4 vLightPos;
    XMFLOAT4 vLightParams;
    XMFLOAT4 vCameraPos; XMFLOAT4 vMaterialParams;
};

// --------------------------------------------------------------------------
// ゲームロジック (変更なし)
// --------------------------------------------------------------------------
class Game {
public:
    XMINT2 m_logicPos; XMFLOAT2 m_drawPos; float m_moveSpeed;
    int m_mapWidth = 10; int m_mapHeight = 10;
    std::vector<std::vector<int>> m_mapData;
    Game() : m_moveSpeed(0.2f) {
        m_mapData = {
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            {1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
            {1, 0, 0, 0, 2, 0, 0, 0, 0, 1},
            {1, 0, 0, 2, 0, 2, 0, 0, 0, 1},
            {1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
            {1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
            {1, 0, 0, 0, 2, 0, 0, 0, 0, 1},
            {1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
            {1, 0, 0, 0, 0, 0, 0, 0, 0, 1},
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        };
        m_logicPos = { 5, 5 }; m_drawPos = { 5.0f, 5.0f };
    }
    void TryMove(int dx, int dy) {
        int nextX = m_logicPos.x + dx; int nextY = m_logicPos.y + dy;
        if (nextX < 0 || nextX >= m_mapWidth || nextY < 0 || nextY >= m_mapHeight) return;
        int targetTile = m_mapData[nextY][nextX];
        if (targetTile == TILE_WALL) return;
        if (targetTile == TILE_BOX) {
            int beyondX = nextX + dx; int beyondY = nextY + dy;
            if (beyondX < 0 || beyondX >= m_mapWidth || beyondY < 0 || beyondY >= m_mapHeight) return;
            if (m_mapData[beyondY][beyondX] == TILE_FLOOR) {
                m_mapData[nextY][nextX] = TILE_FLOOR; m_mapData[beyondY][beyondX] = TILE_BOX;
                m_logicPos.x = nextX; m_logicPos.y = nextY;
            }
        }
        else { m_logicPos.x = nextX; m_logicPos.y = nextY; }
    }
    void Update() {
        static bool keyProcessed = false;
        int dx = 0, dy = 0; bool anyKeyPressed = false;
        if (GetAsyncKeyState(VK_LEFT) & 0x8000) { dx = -1; anyKeyPressed = true; }
        else if (GetAsyncKeyState(VK_RIGHT) & 0x8000) { dx = 1; anyKeyPressed = true; }
        else if (GetAsyncKeyState(VK_UP) & 0x8000) { dy = 1; anyKeyPressed = true; }
        else if (GetAsyncKeyState(VK_DOWN) & 0x8000) { dy = -1; anyKeyPressed = true; }
        if (anyKeyPressed) { if (!keyProcessed) { TryMove(dx, dy); keyProcessed = true; } }
        else { keyProcessed = false; }
        m_drawPos.x = Lerp(m_drawPos.x, (float)m_logicPos.x, m_moveSpeed);
        m_drawPos.y = Lerp(m_drawPos.y, (float)m_logicPos.y, m_moveSpeed);
    }
private:
    float Lerp(float start, float end, float percent) { return start + (end - start) * percent; }
};
Game g_Game;

// --------------------------------------------------------------------------
// Direct3D 初期化 (変更なし)
// --------------------------------------------------------------------------
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
    ID3D11Texture2D* pBackBuffer = nullptr; g_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pBackBuffer); g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_pRenderTargetView); pBackBuffer->Release();
    D3D11_TEXTURE2D_DESC descDepth = { 0 }; descDepth.Width = width; descDepth.Height = height; descDepth.MipLevels = 1; descDepth.ArraySize = 1; descDepth.Format = DXGI_FORMAT_D24_UNORM_S8_UINT; descDepth.SampleDesc.Count = 1; descDepth.Usage = D3D11_USAGE_DEFAULT; descDepth.BindFlags = D3D11_BIND_DEPTH_STENCIL; g_pd3dDevice->CreateTexture2D(&descDepth, nullptr, &g_pDepthStencilBuffer);
    D3D11_DEPTH_STENCIL_VIEW_DESC descDSV = { 0 }; descDSV.Format = descDepth.Format; descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D; g_pd3dDevice->CreateDepthStencilView(g_pDepthStencilBuffer, &descDSV, &g_pDepthStencilView);
    g_pImmediateContext->OMSetRenderTargets(1, &g_pRenderTargetView, g_pDepthStencilView);
    D3D11_VIEWPORT vp; vp.Width = (FLOAT)width; vp.Height = (FLOAT)height; vp.MinDepth = 0.0f; vp.MaxDepth = 1.0f; vp.TopLeftX = 0; vp.TopLeftY = 0; g_pImmediateContext->RSSetViewports(1, &vp);
    ID3DBlob* pVSBlob = nullptr; ID3DBlob* pPSBlob = nullptr;
    D3DCompile(g_shaderHlsl, strlen(g_shaderHlsl), nullptr, nullptr, nullptr, "VS", "vs_4_0", 0, 0, &pVSBlob, nullptr);
    g_pd3dDevice->CreateVertexShader(pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), nullptr, &g_pVertexShader);
    D3DCompile(g_shaderHlsl, strlen(g_shaderHlsl), nullptr, nullptr, nullptr, "PS", "ps_4_0", 0, 0, &pPSBlob, nullptr);
    g_pd3dDevice->CreatePixelShader(pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), nullptr, &g_pPixelShader);
    pPSBlob->Release();
    D3D11_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    g_pd3dDevice->CreateInputLayout(layout, 2, pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), &g_pVertexLayout);
    pVSBlob->Release(); g_pImmediateContext->IASetInputLayout(g_pVertexLayout);
    SimpleVertex vertices[] = {
        { XMFLOAT3(-0.5f, 0.5f, -0.5f), XMFLOAT3(0,1,0) }, { XMFLOAT3(-0.5f, 0.5f, 0.5f), XMFLOAT3(0,1,0) }, { XMFLOAT3(0.5f, 0.5f, -0.5f), XMFLOAT3(0,1,0) },
        { XMFLOAT3(0.5f, 0.5f, -0.5f), XMFLOAT3(0,1,0) }, { XMFLOAT3(-0.5f, 0.5f, 0.5f), XMFLOAT3(0,1,0) }, { XMFLOAT3(0.5f, 0.5f, 0.5f), XMFLOAT3(0,1,0) },
        { XMFLOAT3(-0.5f, -0.5f, -0.5f), XMFLOAT3(0,-1,0) }, { XMFLOAT3(0.5f, -0.5f, -0.5f), XMFLOAT3(0,-1,0) }, { XMFLOAT3(-0.5f, -0.5f, 0.5f), XMFLOAT3(0,-1,0) },
        { XMFLOAT3(-0.5f, -0.5f, 0.5f), XMFLOAT3(0,-1,0) }, { XMFLOAT3(0.5f, -0.5f, -0.5f), XMFLOAT3(0,-1,0) }, { XMFLOAT3(0.5f, -0.5f, 0.5f), XMFLOAT3(0,-1,0) },
        { XMFLOAT3(-0.5f, -0.5f, -0.5f), XMFLOAT3(0,0,-1) }, { XMFLOAT3(-0.5f, 0.5f, -0.5f), XMFLOAT3(0,0,-1) }, { XMFLOAT3(0.5f, -0.5f, -0.5f), XMFLOAT3(0,0,-1) },
        { XMFLOAT3(0.5f, -0.5f, -0.5f), XMFLOAT3(0,0,-1) }, { XMFLOAT3(-0.5f, 0.5f, -0.5f), XMFLOAT3(0,0,-1) }, { XMFLOAT3(0.5f, 0.5f, -0.5f), XMFLOAT3(0,0,-1) },
        { XMFLOAT3(-0.5f, -0.5f, 0.5f), XMFLOAT3(0,0,1) }, { XMFLOAT3(0.5f, -0.5f, 0.5f), XMFLOAT3(0,0,1) }, { XMFLOAT3(-0.5f, 0.5f, 0.5f), XMFLOAT3(0,0,1) },
        { XMFLOAT3(-0.5f, 0.5f, 0.5f), XMFLOAT3(0,0,1) }, { XMFLOAT3(0.5f, -0.5f, 0.5f), XMFLOAT3(0,0,1) }, { XMFLOAT3(0.5f, 0.5f, 0.5f), XMFLOAT3(0,0,1) },
        { XMFLOAT3(-0.5f, -0.5f, 0.5f), XMFLOAT3(-1,0,0) }, { XMFLOAT3(-0.5f, -0.5f, -0.5f), XMFLOAT3(-1,0,0) }, { XMFLOAT3(-0.5f, 0.5f, 0.5f), XMFLOAT3(-1,0,0) },
        { XMFLOAT3(-0.5f, 0.5f, 0.5f), XMFLOAT3(-1,0,0) }, { XMFLOAT3(-0.5f, -0.5f, -0.5f), XMFLOAT3(-1,0,0) }, { XMFLOAT3(-0.5f, 0.5f, -0.5f), XMFLOAT3(-1,0,0) },
        { XMFLOAT3(0.5f, -0.5f, -0.5f), XMFLOAT3(1,0,0) }, { XMFLOAT3(0.5f, -0.5f, 0.5f), XMFLOAT3(1,0,0) }, { XMFLOAT3(0.5f, 0.5f, -0.5f), XMFLOAT3(1,0,0) },
        { XMFLOAT3(0.5f, 0.5f, -0.5f), XMFLOAT3(1,0,0) }, { XMFLOAT3(0.5f, -0.5f, 0.5f), XMFLOAT3(1,0,0) }, { XMFLOAT3(0.5f, 0.5f, 0.5f), XMFLOAT3(1,0,0) },
    };
    D3D11_BUFFER_DESC bd = { 0 }; bd.Usage = D3D11_USAGE_DEFAULT; bd.ByteWidth = sizeof(SimpleVertex) * 36; bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    D3D11_SUBRESOURCE_DATA InitData = { 0 }; InitData.pSysMem = vertices;
    g_pd3dDevice->CreateBuffer(&bd, &InitData, &g_pVertexBuffer);
    UINT stride = sizeof(SimpleVertex); UINT offset = 0; g_pImmediateContext->IASetVertexBuffers(0, 1, &g_pVertexBuffer, &stride, &offset);
    g_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    bd.ByteWidth = sizeof(ConstantBuffer); bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER; g_pd3dDevice->CreateBuffer(&bd, nullptr, &g_pConstantBuffer);
    D3D11_RASTERIZER_DESC rasterDesc = { 0 }; rasterDesc.FillMode = D3D11_FILL_SOLID; rasterDesc.CullMode = D3D11_CULL_BACK;
    g_pd3dDevice->CreateRasterizerState(&rasterDesc, &g_pRasterState); g_pImmediateContext->RSSetState(g_pRasterState);
    return S_OK;
}
void CleanupD3D() {
    if (g_pImmediateContext) g_pImmediateContext->ClearState();
    SAFE_RELEASE(g_pRasterState); SAFE_RELEASE(g_pDepthStencilState); SAFE_RELEASE(g_pConstantBuffer); SAFE_RELEASE(g_pVertexBuffer); SAFE_RELEASE(g_pVertexLayout); SAFE_RELEASE(g_pPixelShader); SAFE_RELEASE(g_pVertexShader); SAFE_RELEASE(g_pDepthStencilView); SAFE_RELEASE(g_pDepthStencilBuffer); SAFE_RELEASE(g_pRenderTargetView); SAFE_RELEASE(g_pSwapChain); SAFE_RELEASE(g_pImmediateContext); SAFE_RELEASE(g_pd3dDevice);
}

// --------------------------------------------------------------------------
// 描画ループ (ライト強度を調整)
// --------------------------------------------------------------------------
void Render() {
    g_Game.Update();
    float ClearColor[4] = { 0.9f, 0.9f, 0.85f, 1.0f };
    g_pImmediateContext->ClearRenderTargetView(g_pRenderTargetView, ClearColor);
    g_pImmediateContext->ClearDepthStencilView(g_pDepthStencilView, D3D11_CLEAR_DEPTH, 1.0f, 0);

    XMVECTOR Eye = XMVectorSet(5.0f, 12.0f, -4.0f, 0.0f);
    XMVECTOR At = XMVectorSet(5.0f, 0.0f, 5.0f, 0.0f);
    XMVECTOR Up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);
    XMMATRIX mView = XMMatrixLookAtLH(Eye, At, Up);
    float aspectRatio = (float)SCREEN_WIDTH / (float)SCREEN_HEIGHT;
    XMMATRIX mProjection = XMMatrixPerspectiveFovLH(XMConvertToRadians(45.0f), aspectRatio, 0.1f, 100.0f);

    ConstantBuffer cb;
    cb.mView = XMMatrixTranspose(mView);
    cb.mProjection = XMMatrixTranspose(mProjection);

    cb.vLightPos = XMFLOAT4(4.5f, 9.0f, 4.5f, 1.0f);
    // 【調整箇所】ライト強度を 1.6 -> 1.3 へ下げる
    cb.vLightParams = XMFLOAT4(25.0f, 1.3f, 0.0f, 0.0f);

    XMStoreFloat4(&cb.vCameraPos, Eye);

    g_pImmediateContext->VSSetShader(g_pVertexShader, nullptr, 0);
    g_pImmediateContext->PSSetShader(g_pPixelShader, nullptr, 0);
    g_pImmediateContext->VSSetConstantBuffers(0, 1, &g_pConstantBuffer);
    g_pImmediateContext->PSSetConstantBuffers(0, 1, &g_pConstantBuffer);

    for (int y = 0; y < g_Game.m_mapHeight; ++y) {
        for (int x = 0; x < g_Game.m_mapWidth; ++x) {
            XMMATRIX mWorld = XMMatrixScaling(1.0f, 0.1f, 1.0f) * XMMatrixTranslation((float)x, -0.55f, (float)y);
            cb.mWorld = XMMatrixTranspose(mWorld);
            cb.vBaseColor = XMFLOAT4(0.85f, 0.8f, 0.75f, 1.0f);
            cb.vMaterialParams = XMFLOAT4(0.3f, 0.2f, 16.0f, 0.0f);
            g_pImmediateContext->UpdateSubresource(g_pConstantBuffer, 0, nullptr, &cb, 0, 0);
            g_pImmediateContext->Draw(36, 0);

            int tile = g_Game.m_mapData[y][x];
            if (tile == TILE_WALL) {
                mWorld = XMMatrixScaling(1.0f, 1.2f, 1.0f) * XMMatrixTranslation((float)x, 0.1f, (float)y);
                cb.mWorld = XMMatrixTranspose(mWorld);
                cb.vBaseColor = XMFLOAT4(0.65f, 0.55f, 0.85f, 1.0f);
                cb.vMaterialParams = XMFLOAT4(0.5f, 0.5f, 32.0f, 0.0f);
                g_pImmediateContext->UpdateSubresource(g_pConstantBuffer, 0, nullptr, &cb, 0, 0);
                g_pImmediateContext->Draw(36, 0);
            }
            else if (tile == TILE_BOX) {
                mWorld = XMMatrixScaling(0.85f, 0.85f, 0.85f) * XMMatrixTranslation((float)x, 0.0f, (float)y);
                cb.mWorld = XMMatrixTranspose(mWorld);
                cb.vBaseColor = XMFLOAT4(0.9f, 0.6f, 0.2f, 1.0f);
                cb.vMaterialParams = XMFLOAT4(0.8f, 0.8f, 64.0f, 0.0f);
                g_pImmediateContext->UpdateSubresource(g_pConstantBuffer, 0, nullptr, &cb, 0, 0);
                g_pImmediateContext->Draw(36, 0);
            }
        }
    }
    XMMATRIX mPlayerWorld = XMMatrixScaling(0.7f, 0.7f, 0.7f) * XMMatrixTranslation(g_Game.m_drawPos.x, 0.0f, g_Game.m_drawPos.y);
    cb.mWorld = XMMatrixTranspose(mPlayerWorld);
    cb.vBaseColor = XMFLOAT4(1.0f, 0.2f, 0.5f, 1.0f);
    cb.vMaterialParams = XMFLOAT4(0.0f, 1.0f, 64.0f, 0.0f);
    g_pImmediateContext->UpdateSubresource(g_pConstantBuffer, 0, nullptr, &cb, 0, 0);
    g_pImmediateContext->Draw(36, 0);
    g_pSwapChain->Present(1, 0);
}

// --------------------------------------------------------------------------
// Win32 メイン
// --------------------------------------------------------------------------
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_KEYDOWN: if (wParam == VK_ESCAPE) DestroyWindow(hWnd); return 0;
    case WM_DESTROY: PostQuitMessage(0); return 0;
    } return DefWindowProc(hWnd, message, wParam, lParam);
}
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, LPWSTR, int nCmdShow) {
    WNDCLASSEX wcex = { sizeof(WNDCLASSEX) }; wcex.style = CS_HREDRAW | CS_VREDRAW; wcex.lpfnWndProc = WndProc; wcex.hInstance = hInstance; wcex.hCursor = LoadCursor(nullptr, IDC_ARROW); wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1); wcex.lpszClassName = L"Sokoban3DFixedLight"; RegisterClassEx(&wcex);
    g_hWnd = CreateWindow(L"Sokoban3DFixedLight", L"DirectX11 Sokoban Fixed Light", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, SCREEN_WIDTH, SCREEN_HEIGHT, nullptr, nullptr, hInstance, nullptr);
    if (!g_hWnd) return FALSE; ShowWindow(g_hWnd, nCmdShow);
    if (FAILED(InitD3D(g_hWnd))) { CleanupD3D(); return 0; }
    MSG msg = { 0 }; while (msg.message != WM_QUIT) { if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessage(&msg); } else { Render(); } }
    CleanupD3D(); return (int)msg.wParam;
}