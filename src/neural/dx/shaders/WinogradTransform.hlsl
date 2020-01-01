#include "WinogradCommon.h"

// fp16/half math seems a bit slow! - at least on nvidia RTX 2060
#if USE_FP16_MATH == 1 && FP16_IO == 1

[numthreads(kWinogradTransformShaderBlockSize, 1, 1)] 
void input_transform_shader_fp16
(    
    uint3 tid : SV_DispatchThreadID
)
{
    const int H = 8, W = 8;
    int c = tid.x % C;
    int n = tid.x / C;
    if (n > N) return;

    float16_t board[8][8];
    
    // read the board (a row at a time)
    [unroll]
    for (int y = 0; y < 8; y++)
    {
        int index = INDEX_NCHW(n, c, y, 0) / 4;
        float16_t4 r1 = input[index];
        float16_t4 r2 = input[index + 1];
        board[y][0] = r1.x;
        board[y][1] = r1.y;
        board[y][2] = r1.z;
        board[y][3] = r1.w;
        board[y][4] = r2.x;
        board[y][5] = r2.y;
        board[y][6] = r2.z;
        board[y][7] = r2.w;
    }

    // top-left
    {
        float16_t inEl[6][6] = {0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0};

        [unroll]
        for (int i = 0; i < 5; i++)
            [unroll]
            for (int j = 0; j < 5; j++)
                inEl[i + 1][j + 1] = board[i][j];

        // ii) transform it
        inputTransform4x4_gpu(inEl, inEl);

        // iii) write to output
        [unroll]
        for (int y = 0; y < 6; y++)
            [unroll]
            for (int x = 0; x < 6; x++)
                transformedInput[TEMP_INDEX_HWNC(y, x, n * 4 + 0, c)] = inEl[y][x];
    }

    // top-right
    {
        float16_t inEl[6][6] = { 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0 };

        [unroll]
        for (int i = 0; i < 5; i++)
            [unroll]
            for (int j = 0; j < 5; j++)
                inEl[i + 1][j] = board[i][j+3];


        // ii) transform it
        inputTransform4x4_gpu(inEl, inEl);

        // iii) write to output
        [unroll]
        for (int y = 0; y < 6; y++)
            [unroll]
            for (int x = 0; x < 6; x++)
                transformedInput[TEMP_INDEX_HWNC(y, x, n * 4 + 1, c)] = inEl[y][x];
    }


    // bottom-left
    {
        float16_t inEl[6][6] = { 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0 };

        [unroll]
        for (int i = 0; i < 5; i++)
            [unroll]
            for (int j = 0; j < 5; j++)
                inEl[i][j + 1] = board[i+3][j];

        // ii) transform it
        inputTransform4x4_gpu(inEl, inEl);
        
        // iii) write to output
        [unroll]
        for (int y = 0; y < 6; y++)
            [unroll]
            for (int x = 0; x < 6; x++)
                transformedInput[TEMP_INDEX_HWNC(y, x, n * 4 + 2, c)] = inEl[y][x];
    }

    // bottom-right
    {
        float16_t inEl[6][6] = { 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0 };

        [unroll]
        for (int i = 0; i < 5; i++)
            [unroll]
            for (int j = 0; j < 5; j++)
                inEl[i][j] = board[i+3][j+3];

        // ii) transform it
        inputTransform4x4_gpu(inEl, inEl);

        // iii) write to output
        [unroll]
        for (int y = 0; y < 6; y++)
            [unroll]
            for (int x = 0; x < 6; x++)
                transformedInput[TEMP_INDEX_HWNC(y, x, n * 4 + 3, c)] = inEl[y][x];
    }
}

[numthreads(kWinogradTransformShaderBlockSize, 1, 1)]
void output_transform_shader_fp16
(
    uint3 tid : SV_DispatchThreadID
)
{
    const int H = 8, W = 8;

    int k = tid.x % C;      // C is set to K in the constant buffer
    int n = tid.x / C;
    if (n > N) return;

    float16_t board[8][8];
    float16_t b = useBias ? bias[k] : 0;

    [unroll]
    for (int hStart = 0; hStart < 8; hStart += 4)
        [unroll]
        for (int wStart = 0; wStart < 8; wStart += 4)
        {
            //  i) read to per thread registers (for doing output transform)
            int shln = n * 4 + (hStart / 4) * 2 + (wStart / 4);
            float16_t outElTransformed[6][6];
            [unroll]
            for (int y = 0; y < 6; y++)
                [unroll]
                for (int x = 0; x < 6; x++)
                    outElTransformed[y][x] = transformedOutput[TEMP_INDEX_HWNC(y, x, shln, k)];

            // ii) transform it
            float16_t outEl[4][4];
            outputTransform4x4_gpu(outEl, outElTransformed);

            {
                [unroll]
                for (int y = 0; y < 4; y++)
                    [unroll]
                    for (int x = 0; x < 4; x++)
                        board[hStart + y][wStart + x] = outEl[y][x];
            }
        }

    // iii) write to output
    {
        [unroll]
        for (int y = 0; y < 8; y++)
        {
            int index = INDEX_NCHW(n, k, y, 0) / 4;
            // can possibly use uint4 to write entire row at a time?
            // couldn't find half2 to uint re-interpret functions :(
            // same issue for reads.
            float16_t4 r1;
            float16_t4 r2;
            r1.x = board[y][0];
            r1.y = board[y][1];
            r1.z = board[y][2];
            r1.w = board[y][3];
            r2.x = board[y][4];
            r2.y = board[y][5];
            r2.z = board[y][6];
            r2.w = board[y][7];

            // bias
            r1 += b;
            r2 += b;

            // residual add
            if (skipAdd) {
                r1 += skipConnection[index];
                r2 += skipConnection[index + 1];
            }

            // relu
            if (relu) {
              float16_t4 zeros = float16_t4(0, 0, 0, 0);
              r1 = max(r1, zeros);
              r2 = max(r2, zeros);
            }
            output[index]     = r1;
            output[index + 1] = r2;
        }
    }
}

#else

//----------------------------- FP32 versions of the same shaders above ------------------------------//

[numthreads(kWinogradTransformShaderBlockSize, 1, 1)]
#if FP16_IO == 1
void input_transform_shader_fp16
#else
void input_transform_shader_fp32
#endif
(
    uint3 tid : SV_DispatchThreadID
)
{
    const int H = 8, W = 8;
    int c = tid.x % C;
    int n = tid.x / C;
    if (n > N) return;

    float board[8][8];

    // read the board (a row at a time)
    [unroll]
    for (int y = 0; y < 8; y++)
    {
        int index = INDEX_NCHW(n, c, y, 0) / 4;
        float4 r1 = input[index];
        float4 r2 = input[index + 1];
        board[y][0] = r1.x;
        board[y][1] = r1.y;
        board[y][2] = r1.z;
        board[y][3] = r1.w;
        board[y][4] = r2.x;
        board[y][5] = r2.y;
        board[y][6] = r2.z;
        board[y][7] = r2.w;
    }

    // top-left
    {
        float inEl[6][6] = { 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0 };

        [unroll]
        for (int i = 0; i < 5; i++)
            [unroll]
            for (int j = 0; j < 5; j++)
                inEl[i + 1][j + 1] = board[i][j];

        // ii) transform it
        inputTransform4x4_gpu(inEl, inEl);

        // iii) write to output
        [unroll]
        for (int y = 0; y < 6; y++)
            [unroll]
            for (int x = 0; x < 6; x++)
                transformedInput[TEMP_INDEX_HWNC(y, x, n * 4 + 0, c)] = inEl[y][x];
    }

    // top-right
    {
        float inEl[6][6] = { 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0 };

        [unroll]
        for (int i = 0; i < 5; i++)
            [unroll]
            for (int j = 0; j < 5; j++)
                inEl[i + 1][j] = board[i][j + 3];


        // ii) transform it
        inputTransform4x4_gpu(inEl, inEl);

        // iii) write to output
        [unroll]
        for (int y = 0; y < 6; y++)
            [unroll]
            for (int x = 0; x < 6; x++)
                transformedInput[TEMP_INDEX_HWNC(y, x, n * 4 + 1, c)] = inEl[y][x];
    }


    // bottom-left
    {
        float inEl[6][6] = { 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0 };

        [unroll]
        for (int i = 0; i < 5; i++)
            [unroll]
            for (int j = 0; j < 5; j++)
                inEl[i][j + 1] = board[i + 3][j];

        // ii) transform it
        inputTransform4x4_gpu(inEl, inEl);

        // iii) write to output
        [unroll]
        for (int y = 0; y < 6; y++)
            [unroll]
        for (int x = 0; x < 6; x++)
            transformedInput[TEMP_INDEX_HWNC(y, x, n * 4 + 2, c)] = inEl[y][x];
    }

    // bottom-right
    {
        float inEl[6][6] = { 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0 };

        [unroll]
        for (int i = 0; i < 5; i++)
            [unroll]
        for (int j = 0; j < 5; j++)
            inEl[i][j] = board[i + 3][j + 3];

        // ii) transform it
        inputTransform4x4_gpu(inEl, inEl);

        // iii) write to output
        [unroll]
        for (int y = 0; y < 6; y++)
            [unroll]
        for (int x = 0; x < 6; x++)
            transformedInput[TEMP_INDEX_HWNC(y, x, n * 4 + 3, c)] = inEl[y][x];
    }
}


[numthreads(kWinogradTransformShaderBlockSize, 1, 1)]
#if FP16_IO == 1
void output_transform_shader_fp16
#else
void output_transform_shader_fp32
#endif
(
    uint3 tid : SV_DispatchThreadID
)
{
    const int H = 8, W = 8;

    int k = tid.x % C;      // C is set to K in the constant buffer
    int n = tid.x / C;
    if (n > N) return;

    float board[8][8];
    float b = useBias ? bias[k] : 0;

    [unroll]
    for (int hStart = 0; hStart < 8; hStart += 4)
        [unroll]
        for (int wStart = 0; wStart < 8; wStart += 4)
        {
            //  i) read to per thread registers (for doing output transform)
            int shln = n * 4 + (hStart / 4) * 2 + (wStart / 4);
            float outElTransformed[6][6];
            [unroll]
            for (int y = 0; y < 6; y++)
                [unroll]
                for (int x = 0; x < 6; x++)
                    outElTransformed[y][x] = transformedOutput[TEMP_INDEX_HWNC(y, x, shln, k)];

            // ii) transform it
            float outEl[4][4];
            outputTransform4x4_gpu(outEl, outElTransformed);

            {
                [unroll]
                for (int y = 0; y < 4; y++)
                    [unroll]
                    for (int x = 0; x < 4; x++)
                        board[hStart + y][wStart + x] = outEl[y][x];
            }
        }

    // iii) write to output
    {
        [unroll]
        for (int y = 0; y < 8; y++)
        {
            int index = INDEX_NCHW(n, k, y, 0) / 4;
            // can possibly use uint4 to write entire row at a time?
            // couldn't find half2 to uint re-interpret functions :(
            // same issue for reads.
            float4 r1;
            float4 r2;
            r1.x = board[y][0];
            r1.y = board[y][1];
            r1.z = board[y][2];
            r1.w = board[y][3];
            r2.x = board[y][4];
            r2.y = board[y][5];
            r2.z = board[y][6];
            r2.w = board[y][7];

            // bias
            r1 += b;
            r2 += b;

            // residual add
            if (skipAdd) {
                r1 += skipConnection[index];
                r2 += skipConnection[index+1];
            }

            // relu
            if (relu) {
              float4 zeros = float4(0, 0, 0, 0);
              r1 = max(r1, zeros);
              r2 = max(r2, zeros);
            }

            output[index]     = r1;
            output[index + 1] = r2;
        }
    }
}

#endif