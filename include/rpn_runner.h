#pragma once

#include <vector>
#include <string>

// 前向声明 lynxi SDK 类型
typedef void* lynContext_t;
typedef void* lynStream_t;
typedef void* lynModel_t;

// RPN 运行器（基于 lynxi SDK）
class RPNRunner {
public:
    RPNRunner(const std::string& model_path);
    ~RPNRunner();
    
    // 运行 RPN 推理
    // 输入: rpn_input_map [1, 64, 496, 432] NCHW float32
    // 输出: box_map [1, 42, 496, 432], score_map [1, 18, 496, 432]
    void run(
        const float* rpn_input_map,
        float* box_map,      // [1, 42, 496, 432]
        float* score_map     // [1, 18, 496, 432]
    );
    
private:
    void cleanup();
    
    void* engine_;      // lynModel_t
    void* stream_;       // lynStream_t
    void* context_;      // lynContext_t
    bool initialized_;
    
    void* dev_input_;   // 设备输入缓冲区
    void* dev_output_;  // 设备输出缓冲区
    float* host_output_; // 主机输出缓冲区
    
    uint64_t input_size_;
    uint64_t output_size_;
};
