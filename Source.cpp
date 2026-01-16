#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "d3dcompiler.lib")

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <vector>
#include <cmath>

// リンク設定: プロジェクトのプロパティで d3d11.lib と d3dcompiler.lib を追加してください。

using namespace DirectX;

// --------------------------------------------------------------------------
// 定数・グローバル変数 (ComPtr未使用のため生ポインタで管理)
// --------------------------------------------------------------------------
const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 600;
const float GRID_SIZE = 1.0f; // 3D空間での1マスの大きさ

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

// 安全な解放マクロ
#define SAFE_RELEASE(p) { if(p) { (p)->Release(); (p)=nullptr; } }

// --------------------------------------------------------------------------
// シェーダーとデータ構造
// --------------------------------------------------------------------------

// 簡易シェーダーコード (hlslを埋め込み)
// 実際はテクスチャ座標(Tex)を受け取り、Texture2Dをサンプリングしますが、
// 今回は簡易化のため色(Color)を直接渡します。
const char* g_shaderHlsl = R"(
cbuffer ConstantBuffer : register(b0) {
    matrix World;
    matrix View;
    matrix Projection;
    float4 Color; // オブジェクトの色
}

struct VS_INPUT {
    float4 Pos : POSITION;
};

struct PS_INPUT {
    float4 Pos : SV_POSITION;
    float4 Color : COLOR;
};

PS_INPUT VS(VS_INPUT input) {
    PS_INPUT output = (PS_INPUT)0;
    output.Pos = mul(input.Pos, World);
    output.Pos = mul(output.Pos, View);
    output.Pos = mul(output.Pos, Projection);
    output.Color = Color; // 定数バッファの色をそのままピクセルシェーダーへ
    return output;
}

float4 PS(PS_INPUT input) : SV_Target {
    return input.Color;
}
)";

// 頂点構造体
struct SimpleVertex {
    XMFLOAT3 Pos;
};

// 定数バッファ構造体 (シェーダー側と合わせる)
struct ConstantBuffer {
    XMMATRIX mWorld;
    XMMATRIX mView;
    XMMATRIX mProjection;
    XMFLOAT4 vColor;
};

// --------------------------------------------------------------------------
// ゲームロジック用クラス
// --------------------------------------------------------------------------
class Game {
public:
    XMINT2 m_logicPos; // 論理グリッド座標 (例: x=5, y=3)
    XMFLOAT2 m_drawPos; // 描画用物理座標 (例: x=5.12f, y=3.00f)
    float m_moveSpeed;  // 移動アニメーションの速度

    Game() : m_logicPos(5, 5), m_drawPos(5.0f, 5.0f), m_moveSpeed(0.15f) {}

    void Update() {
        // 簡易入力処理 (矢印キー)
        // 本来はメッセージループや入力管理クラスで行うべきですが、サンプルとしてここに記述
        static bool keyProcessed = false;
        int dx = 0, dy = 0;
        if (GetAsyncKeyState(VK_LEFT) & 0x8000) dx = -1;
        else if (GetAsyncKeyState(VK_RIGHT) & 0x8000) dx = 1;
        else if (GetAsyncKeyState(VK_UP) & 0x8000) dy = 1; // 3D空間ではZ奥がプラスと仮定
        else if (GetAsyncKeyState(VK_DOWN) & 0x8000) dy = -1;
        else keyProcessed = false;

        // キーが押されていて、まだ処理していない場合
        if ((dx != 0 || dy != 0) && !keyProcessed) {
            // ここに倉庫番の移動可否判定が入ります (今回は無条件移動)
            m_logicPos.x += dx;
            m_logicPos.y += dy;
            keyProcessed = true;
        }

        // ★滑らかな移動ロジックの中核★
        // 現在の描画位置を、目標となる論理位置へ少しずつ近づける (線形補間)
        // targetX = m_logicPos.x * GRID_SIZE;
        m_drawPos.x = Lerp(m_drawPos.x, (float)m_logicPos.x, m_moveSpeed);
        m_drawPos.y = Lerp(m_drawPos.y, (float)m_logicPos.y, m_moveSpeed);
    }

private:
    // 線形補間ヘルパー関数
    float Lerp(float start, float end, float percent) {
        return start + (end - start) * percent;
    }
};
Game g_Game;


// --------------------------------------------------------------------------
// Direct3D 初期化とクリーンアップ
// --------------------------------------------------------------------------
HRESULT InitD3D(HWND hWnd) {
    HRESULT hr = S_OK;

    RECT rc;
    GetClientRect(hWnd, &rc);
    UINT width = rc.right - rc.left;
    UINT height = rc.bottom - rc.top;

    UINT createDeviceFlags = 0;
#ifdef _DEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    D3D_FEATURE_LEVEL featureLevels[] = { D3D_FEATURE_LEVEL_11_0 };
    D3D_FEATURE_LEVEL featureLevel;

    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.BufferCount = 1;
    sd.BufferDesc.Width = width;
    sd.BufferDesc.Height = height;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;

    hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags,
        featureLevels, 1, D3D11_SDK_VERSION, &sd, &g_pSwapChain, &g_pd3dDevice, &featureLevel, &g_pImmediateContext);
    if (FAILED(hr)) return hr;

    // Render Target View作成
    ID3D11Texture2D* pBackBuffer = nullptr;
    hr = g_pSwapChain->GetBuffer(0, __uuidof(ID3D11Texture2D), (LPVOID*)&pBackBuffer);
    if (FAILED(hr)) return hr;
    hr = g_pd3dDevice->CreateRenderTargetView(pBackBuffer, nullptr, &g_pRenderTargetView);
    pBackBuffer->Release();
    if (FAILED(hr)) return hr;

    // Depth Stencil Texture作成 (3D描画の前後関係判定に必要)
    D3D11_TEXTURE2D_DESC descDepth;
    ZeroMemory(&descDepth, sizeof(descDepth));
    descDepth.Width = width;
    descDepth.Height = height;
    descDepth.MipLevels = 1;
    descDepth.ArraySize = 1;
    descDepth.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    descDepth.SampleDesc.Count = 1;
    descDepth.SampleDesc.Quality = 0;
    descDepth.Usage = D3D11_USAGE_DEFAULT;
    descDepth.BindFlags = D3D11_BIND_DEPTH_STENCIL;
    descDepth.CPUAccessFlags = 0;
    descDepth.MiscFlags = 0;
    hr = g_pd3dDevice->CreateTexture2D(&descDepth, nullptr, &g_pDepthStencilBuffer);
    if (FAILED(hr)) return hr;

    // Depth Stencil View作成
    D3D11_DEPTH_STENCIL_VIEW_DESC descDSV;
    ZeroMemory(&descDSV, sizeof(descDSV));
    descDSV.Format = descDepth.Format;
    descDSV.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
    descDSV.Texture2D.MipSlice = 0;
    hr = g_pd3dDevice->CreateDepthStencilView(g_pDepthStencilBuffer, &descDSV, &g_pDepthStencilView);
    if (FAILED(hr)) return hr;

    g_pImmediateContext->OMSetRenderTargets(1, &g_pRenderTargetView, g_pDepthStencilView);

    // Viewport設定
    D3D11_VIEWPORT vp;
    vp.Width = (FLOAT)width;
    vp.Height = (FLOAT)height;
    vp.MinDepth = 0.0f;
    vp.MaxDepth = 1.0f;
    vp.TopLeftX = 0;
    vp.TopLeftY = 0;
    g_pImmediateContext->RSSetViewports(1, &vp);

    // シェーダーのコンパイルと作成
    ID3DBlob* pVSBlob = nullptr;
    hr = D3DCompile(g_shaderHlsl, strlen(g_shaderHlsl), nullptr, nullptr, nullptr, "VS", "vs_4_0", 0, 0, &pVSBlob, nullptr);
    if (FAILED(hr)) return hr;
    hr = g_pd3dDevice->CreateVertexShader(pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), nullptr, &g_pVertexShader);
    if (FAILED(hr)) { pVSBlob->Release(); return hr; }

    ID3DBlob* pPSBlob = nullptr;
    hr = D3DCompile(g_shaderHlsl, strlen(g_shaderHlsl), nullptr, nullptr, nullptr, "PS", "ps_4_0", 0, 0, &pPSBlob, nullptr);
    if (FAILED(hr)) return hr;
    hr = g_pd3dDevice->CreatePixelShader(pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), nullptr, &g_pPixelShader);
    pPSBlob->Release();
    if (FAILED(hr)) return hr;

    // インプットレイアウト作成
    D3D11_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    UINT numElements = ARRAYSIZE(layout);
    hr = g_pd3dDevice->CreateInputLayout(layout, numElements, pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), &g_pVertexLayout);
    pVSBlob->Release();
    if (FAILED(hr)) return hr;
    g_pImmediateContext->IASetInputLayout(g_pVertexLayout);

    // 頂点バッファ作成 (単純な1x1の四角形)
    // 3D空間でXZ平面に寝かせた四角形を定義します
    SimpleVertex vertices[] = {
        { XMFLOAT3(-0.5f, 0.0f, -0.5f) }, // 左下
        { XMFLOAT3(-0.5f, 0.0f,  0.5f) }, // 左上
        { XMFLOAT3(0.5f, 0.0f, -0.5f) }, // 右下
        { XMFLOAT3(0.5f, 0.0f,  0.5f) }, // 右上
    };
    D3D11_BUFFER_DESC bd;
    ZeroMemory(&bd, sizeof(bd));
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.ByteWidth = sizeof(SimpleVertex) * 4;
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bd.CPUAccessFlags = 0;
    D3D11_SUBRESOURCE_DATA InitData;
    ZeroMemory(&InitData, sizeof(InitData));
    InitData.pSysMem = vertices;
    hr = g_pd3dDevice->CreateBuffer(&bd, &InitData, &g_pVertexBuffer);
    if (FAILED(hr)) return hr;

    UINT stride = sizeof(SimpleVertex);
    UINT offset = 0;
    g_pImmediateContext->IASetVertexBuffers(0, 1, &g_pVertexBuffer, &stride, &offset);
    g_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

    // 定数バッファ作成
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.ByteWidth = sizeof(ConstantBuffer);
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bd.CPUAccessFlags = 0;
    hr = g_pd3dDevice->CreateBuffer(&bd, nullptr, &g_pConstantBuffer);
    if (FAILED(hr)) return hr;

    // ラスタライザステート (カリングなし設定など)
    D3D11_RASTERIZER_DESC rasterDesc;
    ZeroMemory(&rasterDesc, sizeof(rasterDesc));
    rasterDesc.FillMode = D3D11_FILL_SOLID;
    rasterDesc.CullMode = D3D11_CULL_NONE; // 裏面も描画する
    hr = g_pd3dDevice->CreateRasterizerState(&rasterDesc, &g_pRasterState);
    if (FAILED(hr)) return hr;
    g_pImmediateContext->RSSetState(g_pRasterState);

    return S_OK;
}

void CleanupD3D() {
    if (g_pImmediateContext) g_pImmediateContext->ClearState();
    SAFE_RELEASE(g_pRasterState);
    SAFE_RELEASE(g_pDepthStencilState);
    SAFE_RELEASE(g_pConstantBuffer);
    SAFE_RELEASE(g_pVertexBuffer);
    SAFE_RELEASE(g_pVertexLayout);
    SAFE_RELEASE(g_pPixelShader);
    SAFE_RELEASE(g_pVertexShader);
    SAFE_RELEASE(g_pDepthStencilView);
    SAFE_RELEASE(g_pDepthStencilBuffer);
    SAFE_RELEASE(g_pRenderTargetView);
    SAFE_RELEASE(g_pSwapChain);
    SAFE_RELEASE(g_pImmediateContext);
    SAFE_RELEASE(g_pd3dDevice);
}

// --------------------------------------------------------------------------
// 描画ループ
// --------------------------------------------------------------------------
void Render() {
    g_Game.Update(); // ゲームロジック更新

    // 画面クリア (背景色)
    float ClearColor[4] = { 0.1f, 0.1f, 0.1f, 1.0f };
    g_pImmediateContext->ClearRenderTargetView(g_pRenderTargetView, ClearColor);
    g_pImmediateContext->ClearDepthStencilView(g_pDepthStencilView, D3D11_CLEAR_DEPTH, 1.0f, 0);

    // カメラ行列の計算 (真上から見下ろす)
    XMVECTOR Eye = XMVectorSet(5.0f, 10.0f, 5.0f, 0.0f); // カメラ位置 (上空)
    XMVECTOR At = XMVectorSet(5.0f, 0.0f, 5.0f, 0.0f);   // 注視点 (地表)
    XMVECTOR Up = XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f);   // 上方向 (Z軸奥を上とする)
    XMMATRIX mView = XMMatrixLookAtLH(Eye, At, Up);

    // 射影行列の計算 (正射影：パースがつかない)
    // 幅12グリッド分が表示されるように設定
    float viewWidth = 12.0f * GRID_SIZE;
    float viewHeight = viewWidth * ((float)SCREEN_HEIGHT / SCREEN_WIDTH);
    XMMATRIX mProjection = XMMatrixOrthographicLH(viewWidth, viewHeight, 0.1f, 100.0f);

    ConstantBuffer cb;
    cb.mView = XMMatrixTranspose(mView);
    cb.mProjection = XMMatrixTranspose(mProjection);

    g_pImmediateContext->VSSetShader(g_pVertexShader, nullptr, 0);
    g_pImmediateContext->PSSetShader(g_pPixelShader, nullptr, 0);
    g_pImmediateContext->VSSetConstantBuffers(0, 1, &g_pConstantBuffer);
    g_pImmediateContext->PSSetConstantBuffers(0, 1, &g_pConstantBuffer);

    // --- 床の描画 (10x10のグリッド) ---
    cb.vColor = XMFLOAT4(0.5f, 0.5f, 0.5f, 1.0f); // 灰色
    for (int z = 0; z < 10; ++z) {
        for (int x = 0; x < 10; ++x) {
            // 各タイルの位置にワールド行列を設定
            XMMATRIX mWorld = XMMatrixTranslation((float)x * GRID_SIZE, 0.0f, (float)z * GRID_SIZE);
            cb.mWorld = XMMatrixTranspose(mWorld);
            g_pImmediateContext->UpdateSubresource(g_pConstantBuffer, 0, nullptr, &cb, 0, 0);
            g_pImmediateContext->Draw(4, 0);
        }
    }

    // --- プレイヤーの描画 (滑らかな位置) ---
    // プレイヤーは少し浮かせて(Y=0.01)、床と重ならないようにする
    XMMATRIX mPlayerWorld = XMMatrixTranslation(g_Game.m_drawPos.x * GRID_SIZE, 0.01f, g_Game.m_drawPos.y * GRID_SIZE);
    cb.mWorld = XMMatrixTranspose(mPlayerWorld);
    cb.vColor = XMFLOAT4(1.0f, 0.2f, 0.2f, 1.0f); // 赤色
    g_pImmediateContext->UpdateSubresource(g_pConstantBuffer, 0, nullptr, &cb, 0, 0);
    g_pImmediateContext->Draw(4, 0);

    g_pSwapChain->Present(1, 0); // VSync有効
}

// --------------------------------------------------------------------------
// Win32 ウィンドウプロシージャとメイン関数
// --------------------------------------------------------------------------
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
    switch (message) {
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE) DestroyWindow(hWnd);
        return 0;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hWnd, message, wParam, lParam);
}

int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, LPWSTR, int nCmdShow) {
    WNDCLASSEX wcex = { sizeof(WNDCLASSEX) };
    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = WndProc;
    wcex.hInstance = hInstance;
    wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wcex.lpszClassName = L"Sokoban3DClass";
    RegisterClassEx(&wcex);

    g_hWnd = CreateWindow(L"Sokoban3DClass", L"DirectX11 Sokoban Base", WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, SCREEN_WIDTH, SCREEN_HEIGHT, nullptr, nullptr, hInstance, nullptr);
    if (!g_hWnd) return FALSE;

    ShowWindow(g_hWnd, nCmdShow);
    if (FAILED(InitD3D(g_hWnd))) { CleanupD3D(); return 0; }

    // メインループ
    MSG msg = { 0 };
    while (msg.message != WM_QUIT) {
        if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        else {
            Render();
        }
    }

    CleanupD3D();
    return (int)msg.wParam;
}