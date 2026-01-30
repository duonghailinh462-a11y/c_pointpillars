#include "rpn_runner.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <fstream>
#include <stdexcept>

// 包含 lynxi SDK 头文件
#include <lyn_api.h>

RPNRunner::RPNRunner(const std::string& model_path) 
    : engine_(nullptr), initialized_(false) {
    
    // 1. 创建 Context（如果还没有创建的话，这里假设全局已创建）
    // 注意：通常 context 应该在 main 函数中创建一次，这里为了简化先检查
    lynContext_t ctx = nullptr;
    lynError_t err = lynCreateContext(&ctx, 0);  // chipNum = 0 表示使用默认芯片
    if (err != 0) {  // 0 表示成功
        std::cerr << "RPNRunner: 创建 Context 失败，错误码: " << err << std::endl;
        throw std::runtime_error("Failed to create lynxi context");
    }
    
    // 2. 创建 Stream
    lynStream_t stream = nullptr;
    err = lynCreateStream(&stream);
    if (err != 0) {
        std::cerr << "RPNRunner: 创建 Stream 失败，错误码: " << err << std::endl;
        lynDestroyContext(ctx);
        throw std::runtime_error("Failed to create lynxi stream");
    }
    
    // 3. 加载模型
    lynModel_t model = nullptr;
    err = lynLoadModel(model_path.c_str(), &model);
    if (err != 0) {
        std::cerr << "RPNRunner: 加载模型失败: " << model_path << ", 错误码: " << err << std::endl;
        lynDestroyStream(stream);
        lynDestroyContext(ctx);
        throw std::runtime_error("Failed to load RPN model");
    }
    
    engine_ = model;  // 保存模型句柄
    stream_ = stream;
    context_ = ctx;
    
    // 4. 获取模型输入输出大小
    err = lynModelGetInputDataTotalLen(model, &input_size_);
    if (err != 0) {
        std::cerr << "RPNRunner: 获取输入大小失败" << std::endl;
        cleanup();
        throw std::runtime_error("Failed to get input size");
    }
    
    err = lynModelGetOutputDataTotalLen(model, &output_size_);
    if (err != 0) {
        std::cerr << "RPNRunner: 获取输出大小失败" << std::endl;
        cleanup();
        throw std::runtime_error("Failed to get output size");
    }
    
    // 5. 分配设备内存
    err = lynMalloc((void**)&dev_input_, input_size_);
    if (err != 0) {
        std::cerr << "RPNRunner: 分配输入内存失败" << std::endl;
        cleanup();
        throw std::runtime_error("Failed to allocate input memory");
    }
    
    err = lynMalloc((void**)&dev_output_, output_size_);
    if (err != 0) {
        std::cerr << "RPNRunner: 分配输出内存失败" << std::endl;
        cleanup();
        throw std::runtime_error("Failed to allocate output memory");
    }
    
    // 6. 分配主机输出缓冲区
    host_output_ = (float*)malloc(output_size_);
    if (!host_output_) {
        cleanup();
        throw std::runtime_error("Failed to allocate host output buffer");
    }
    
    initialized_ = true;
    std::cout << "RPNRunner: 模型加载成功" << std::endl;
    std::cout << "  输入大小: " << input_size_ << " 字节" << std::endl;
    std::cout << "  输出大小: " << output_size_ << " 字节" << std::endl;
}

RPNRunner::~RPNRunner() {
    cleanup();
}

void RPNRunner::cleanup() {
    if (host_output_) {
        free(host_output_);
        host_output_ = nullptr;
    }
    if (dev_output_ && engine_) {
        lynFree(dev_output_);
        dev_output_ = nullptr;
    }
    if (dev_input_ && engine_) {
        lynFree(dev_input_);
        dev_input_ = nullptr;
    }
    if (engine_) {
        lynUnloadModel((lynModel_t)engine_);
        engine_ = nullptr;
    }
    if (stream_) {
        lynDestroyStream((lynStream_t)stream_);
        stream_ = nullptr;
    }
    if (context_) {
        lynDestroyContext((lynContext_t)context_);
        context_ = nullptr;
    }
}

void RPNRunner::run(
    const float* rpn_input_map,
    float* box_map,
    float* score_map) {
    
    if (!initialized_) {
        throw std::runtime_error("RPNRunner not initialized");
    }
    
    lynStream_t stream = (lynStream_t)stream_;
    lynModel_t model = (lynModel_t)engine_;
    
    // 1. 将输入数据拷贝到设备（同步方式，也可以异步）
    // 注意：rpn_input_map 是主机内存，需要拷贝到设备
    lynError_t err = lynMemcpyAsync(stream, dev_input_, (void*)rpn_input_map, input_size_, 
                                    ClientToServer);
    if (err != 0) {
        throw std::runtime_error("Failed to copy input to device");
    }
    
    // 2. 执行推理（异步）
    err = lynExecuteModelAsync(stream, model, dev_input_, dev_output_, 1);  // batchSize = 1
    if (err != 0) {
        throw std::runtime_error("Failed to execute model");
    }
    
    // 3. 将输出数据拷贝回主机（异步）
    err = lynMemcpyAsync(stream, host_output_, dev_output_, output_size_, ServerToClient);
    if (err != 0) {
        throw std::runtime_error("Failed to copy output from device");
    }
    
    // 4. 同步等待所有操作完成
    err = lynSynchronizeStream(stream);
    if (err != 0) {
        throw std::runtime_error("Failed to synchronize stream");
    }
    
    // 5. 解析输出：需要根据实际模型输出格式调整
    // 注意：lynxi SDK的输出可能是多个tensor，需要分别获取
    uint32_t output_tensor_num = 0;
    lynError_t err2 = lynModelGetOutputTensorNum((lynModel_t)engine_, &output_tensor_num);
    if (err2 != 0) {
        std::cerr << "警告: 无法获取输出tensor数量，使用默认解析方式" << std::endl;
    } else {
        std::cout << "  RPN输出tensor数量: " << output_tensor_num << std::endl;
    }
    
    // 如果模型有多个输出tensor，需要分别处理
    // 这里假设：tensor[0] = box_map, tensor[1] = score_map
    const size_t box_map_size = 1 * 42 * 496 * 432 * sizeof(float);
    const size_t score_map_size = 1 * 18 * 496 * 432 * sizeof(float);
    
    if (output_tensor_num >= 2) {
        // 多tensor输出：分别获取每个tensor的数据
        uint64_t box_tensor_size = 0, score_tensor_size = 0;
        lynModelGetOutputTensorDataLenByIndex((lynModel_t)engine_, 0, &box_tensor_size);
        lynModelGetOutputTensorDataLenByIndex((lynModel_t)engine_, 1, &score_tensor_size);
        
        std::cout << "  Box tensor大小: " << box_tensor_size << " 字节" << std::endl;
        std::cout << "  Score tensor大小: " << score_tensor_size << " 字节" << std::endl;
        
        // 计算tensor在输出缓冲区中的偏移
        uint64_t offset0 = 0;
        uint64_t offset1 = box_tensor_size;
        
        std::memcpy(box_map, (char*)host_output_ + offset0, 
                   std::min(box_map_size, static_cast<size_t>(box_tensor_size)));
        std::memcpy(score_map, (char*)host_output_ + offset1, 
                   std::min(score_map_size, static_cast<size_t>(score_tensor_size)));
    } else {
        // 单tensor输出：假设连续排列 [box_map, score_map]
        if (output_size_ < (box_map_size + score_map_size)) {
            std::cerr << "警告: 输出大小不匹配，期望: " << (box_map_size + score_map_size) 
                      << ", 实际: " << output_size_ << std::endl;
        }
        
        std::memcpy(box_map, host_output_, std::min(box_map_size, output_size_));
        if (output_size_ >= box_map_size + score_map_size) {
            std::memcpy(score_map, (char*)host_output_ + box_map_size, score_map_size);
        } else {
            std::memset(score_map, 0, score_map_size);
        }
    }
    
    // 打印tensor详细信息用于调试
    if (output_tensor_num >= 1) {
        uint32_t dims[16];
        uint32_t dim_count = 0;
        char tensor_name[128];
        lynDataType_t dtype;
        uint32_t data_num = 0;
        
        for (uint32_t t = 0; t < output_tensor_num && t < 2; ++t) {
            lynModelGetOutputTensorDimsByIndex((lynModel_t)engine_, t, dims, &dim_count);
            lynModelGetOutputTensorNameByIndex((lynModel_t)engine_, t, tensor_name);
            lynModelGetOutputTensorDataTypeByIndex((lynModel_t)engine_, t, &dtype);
            lynModelGetOutputTensorDataNumByIndex((lynModel_t)engine_, t, &data_num);
            
            std::cout << "  Tensor[" << t << "]: name=" << tensor_name 
                      << ", dtype=" << dtype << ", dims=[";
            for (uint32_t d = 0; d < dim_count; ++d) {
                if (d > 0) std::cout << ", ";
                std::cout << dims[d];
            }
            std::cout << "], data_num=" << data_num << std::endl;
        }
    }
    
    // 数据有效性检查（放宽范围，因为回归值可能较大）
    float max_box = 0, max_score = 0;
    int nan_count_box = 0, inf_count_box = 0;
    int nan_count_score = 0, inf_count_score = 0;
    
    for (size_t i = 0; i < box_map_size / sizeof(float); ++i) {
        float val = box_map[i];
        if (!std::isfinite(val)) {
            if (std::isnan(val)) nan_count_box++;
            if (std::isinf(val)) inf_count_box++;
        } else {
            max_box = std::max(max_box, std::abs(val));
        }
    }
    for (size_t i = 0; i < score_map_size / sizeof(float); ++i) {
        float val = score_map[i];
        if (!std::isfinite(val)) {
            if (std::isnan(val)) nan_count_score++;
            if (std::isinf(val)) inf_count_score++;
        } else {
            max_score = std::max(max_score, std::abs(val));
        }
    }
    
    std::cout << "  数据统计: box_max=" << max_box 
              << " (NaN:" << nan_count_box << ", Inf:" << inf_count_box << ")"
              << ", score_max=" << max_score
              << " (NaN:" << nan_count_score << ", Inf:" << inf_count_score << ")" << std::endl;
    
    // 只检查NaN和Inf，不检查数值范围（因为回归值可能很大）
    if (nan_count_box > 0 || inf_count_box > 0 || nan_count_score > 0 || inf_count_score > 0) {
        std::cerr << "警告: RPN输出包含NaN或Inf值，可能有问题" << std::endl;
    }
}
