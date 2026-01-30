#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "voxelizer.h"
#include "pfn.hpp"
#include "rpn_runner.h"
#include "postprocess.h"

// 读取二进制文件
std::vector<float> load_bin(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("无法打开文件: " + std::string(path));
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> buffer(size / sizeof(float));
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return buffer;
}

// 读取点云文件（KITTI格式：每点4个float: x, y, z, intensity）
std::vector<float> load_pointcloud(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("无法打开点云文件: " + path);
    }
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<float> points(size / sizeof(float));
    file.read(reinterpret_cast<char*>(points.data()), size);
    
    std::cout << "✓ 加载点云: " << points.size() / 4 << " 个点" << std::endl;
    return points;
}

void print_boxes(const std::vector<Box3D>& boxes) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "检测结果" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "检测框数量: " << boxes.size() << std::endl;
    
    if (boxes.size() > 0) {
        std::cout << "\n前10个检测框:" << std::endl;
        std::cout << "ID | Score    | Label | [x, y, z, w, l, h, rot]" << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        int n = std::min(10, static_cast<int>(boxes.size()));
        for (int i = 0; i < n; ++i) {
            const auto& b = boxes[i];
            printf("%2d | %.4f | %5d | [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n",
                   i, b.score, b.label, b.x, b.y, b.z, b.w, b.l, b.h, b.rot);
        }
    }
    std::cout << std::string(80, '=') << std::endl;
}

int main(int argc, char* argv[]) {
    // 获取项目根目录（相对于 build 目录的上一级）
    std::string project_root = "..";
    if (!std::filesystem::exists(project_root + "/pfn_weight.bin")) {
        project_root = ".";  // 如果从项目根目录运行，直接使用当前目录
    }
    
    std::string pointcloud_file = project_root + "/test/kitti_000008.bin";
    std::string pfn_weight = project_root + "/pfn_weight.bin";
    std::string pfn_bias = project_root + "/pfn_bias.bin";
    std::string rpn_model = project_root + "/rpn_lynxi/Net_0/apu_0/apu_x/lyn__2026-01-28-11-13-55-749707.mdl";  // 默认路径
    float score_thr = 0.3f;
    float nms_thr = 0.01f;
    int max_num = 100;
    
    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--pointcloud" && i + 1 < argc) {
            pointcloud_file = argv[++i];
        } else if (arg == "--pfn-weight" && i + 1 < argc) {
            pfn_weight = argv[++i];
        } else if (arg == "--pfn-bias" && i + 1 < argc) {
            pfn_bias = argv[++i];
        } else if (arg == "--rpn-model" && i + 1 < argc) {
            rpn_model = argv[++i];
        } else if (arg == "--score-thr" && i + 1 < argc) {
            score_thr = std::stof(argv[++i]);
        } else if (arg == "--nms-thr" && i + 1 < argc) {
            nms_thr = std::stof(argv[++i]);
        } else if (arg == "--max-num" && i + 1 < argc) {
            max_num = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "用法: " << argv[0] << " [选项]\n"
                      << "选项:\n"
                      << "  --pointcloud <path>    点云文件 (默认: test/kitti_000008.bin)\n"
                      << "  --pfn-weight <path>    PFN权重 (默认: pfn_weight.bin)\n"
                      << "  --pfn-bias <path>      PFN偏置 (默认: pfn_bias.bin)\n"
                      << "  --rpn-model <path>     RPN模型路径\n"
                      << "  --score-thr <float>   分数阈值 (默认: 0.3)\n"
                      << "  --nms-thr <float>     NMS阈值 (默认: 0.01)\n"
                      << "  --max-num <int>       最大检测数 (默认: 100)\n";
            return 0;
        }
    }
    
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "PointPillars 完整推理流程" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    try {
        auto total_start = std::chrono::high_resolution_clock::now();
        
        // === 1. 加载点云 ===
        std::cout << "\n--- 步骤1: 加载点云 ---" << std::endl;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto points = load_pointcloud(pointcloud_file);
        auto t1 = std::chrono::high_resolution_clock::now();
        double load_time = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "耗时: " << std::fixed << std::setprecision(2) << load_time << " ms" << std::endl;
        
        // === 2. 体素化 ===
        std::cout << "\n--- 步骤2: 体素化 ---" << std::endl;
        t0 = std::chrono::high_resolution_clock::now();
        VoxelConfig voxel_config;
        Voxelizer voxelizer(voxel_config);
        auto voxel_data = voxelizer.generate(points);
        t1 = std::chrono::high_resolution_clock::now();
        double voxel_time = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "体素数: " << voxel_data.num_voxels << std::endl;
        std::cout << "耗时: " << std::fixed << std::setprecision(2) << voxel_time << " ms" << std::endl;
        
        // === 3. 初始化 PFN ===
        std::cout << "\n--- 步骤3: 初始化 PFN (CPU) ---" << std::endl;
        t0 = std::chrono::high_resolution_clock::now();
        PFN_CPU pfn_runner;
        pfn_runner.pfn_weights = load_bin(pfn_weight.c_str());
        pfn_runner.pfn_bias = load_bin(pfn_bias.c_str());
        t1 = std::chrono::high_resolution_clock::now();
        double pfn_init_time = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "PFN权重大小: " << pfn_runner.pfn_weights.size() << std::endl;
        std::cout << "PFN偏置大小: " << pfn_runner.pfn_bias.size() << std::endl;
        std::cout << "耗时: " << std::fixed << std::setprecision(2) << pfn_init_time << " ms" << std::endl;
        
        // === 4. PFN 前向 + Scatter ===
        std::cout << "\n--- 步骤4: PFN 前向 + Scatter ---" << std::endl;
        t0 = std::chrono::high_resolution_clock::now();
        std::vector<float> rpn_input_map(1 * 64 * 496 * 432, 0.0f);
        
        // 转换 VoxelData 到 VoxelInfo
        VoxelInfo voxel_info;
        voxel_info.voxels = voxel_data.voxels.data();
        voxel_info.coordinates = voxel_data.coordinates.data();
        voxel_info.num_points = voxel_data.num_points.data();
        voxel_info.num_voxels = voxel_data.num_voxels;
        voxel_info.max_points = voxel_config.max_num_points;
        
        pfn_runner.run(voxel_info, rpn_input_map.data());
        t1 = std::chrono::high_resolution_clock::now();
        double pfn_time = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "RPN输入形状: [1, 64, 496, 432]" << std::endl;
        std::cout << "耗时: " << std::fixed << std::setprecision(2) << pfn_time << " ms" << std::endl;
        
        // === 5. RPN 推理 (NPU) ===
        std::cout << "\n--- 步骤5: RPN 推理 (NPU) ---" << std::endl;
        t0 = std::chrono::high_resolution_clock::now();
        RPNRunner rpn_runner(rpn_model);
        std::vector<float> box_map(1 * 42 * 496 * 432, 0.0f);   // 6 anchors * 7
        std::vector<float> score_map(1 * 18 * 496 * 432, 0.0f); // 6 anchors * 3
        rpn_runner.run(rpn_input_map.data(), box_map.data(), score_map.data());
        t1 = std::chrono::high_resolution_clock::now();
        double rpn_time = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "耗时: " << std::fixed << std::setprecision(2) << rpn_time << " ms" << std::endl;
        
        // === 6. Decode + NMS ===
        std::cout << "\n--- 步骤6: Anchor Decode + NMS ---" << std::endl;
        t0 = std::chrono::high_resolution_clock::now();
        
        // 检查RPN输出数据有效性（只检查NaN/Inf，不限制数值范围）
        float max_box_val = 0, max_score_val = 0;
        int nan_box = 0, inf_box = 0, nan_score = 0, inf_score = 0;
        
        for (size_t i = 0; i < box_map.size(); ++i) {
            float val = box_map[i];
            if (std::isnan(val)) {
                nan_box++;
            } else if (std::isinf(val)) {
                inf_box++;
            } else if (std::isfinite(val)) {
                max_box_val = std::max(max_box_val, std::abs(val));
            }
        }
        for (size_t i = 0; i < score_map.size(); ++i) {
            float val = score_map[i];
            if (std::isnan(val)) {
                nan_score++;
            } else if (std::isinf(val)) {
                inf_score++;
            } else if (std::isfinite(val)) {
                max_score_val = std::max(max_score_val, std::abs(val));
            }
        }
        
        std::cout << "  RPN输出检查: box_map最大绝对值=" << max_box_val 
                  << " (NaN:" << nan_box << ", Inf:" << inf_box << ")"
                  << ", score_map最大绝对值=" << max_score_val 
                  << " (NaN:" << nan_score << ", Inf:" << inf_score << ")" << std::endl;
        
        // 只检查NaN和Inf，不检查数值范围（回归值可能很大是正常的）
        if (nan_box > 0 || inf_box > 0 || nan_score > 0 || inf_score > 0) {
            std::cerr << "  错误: RPN输出包含NaN或Inf值，无法继续decode！" << std::endl;
            std::cerr << "  请检查：1) RPN模型输出格式 2) 内存布局是否正确" << std::endl;
            return 1;
        }
        
        // 如果值太大，给出警告但不阻止
        if (max_box_val > 1e6 || max_score_val > 1e6) {
            std::cerr << "  警告: RPN输出值较大，可能影响精度" << std::endl;
        }
        
        DecodeConfig decode_cfg;
        decode_cfg.num_classes = 3;  // 6 anchors * 3 classes = 18 channels
        AnchorDecoder decoder(decode_cfg);
        std::cout << "  开始Decode..." << std::endl;
        std::cout.flush();  // 强制刷新输出
        auto decoded = decoder.decode(box_map.data(), score_map.data(), score_thr);
        std::cout << "  开始NMS..." << std::endl;
        std::cout.flush();
        auto final_boxes = nms_bev_rotated(decoded, nms_thr, max_num);
        t1 = std::chrono::high_resolution_clock::now();
        double decode_time = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Decode后: " << decoded.size() << " 个候选框" << std::endl;
        std::cout << "NMS后: " << final_boxes.size() << " 个最终框" << std::endl;
        std::cout << "耗时: " << std::fixed << std::setprecision(2) << decode_time << " ms" << std::endl;
        
        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
        
        // 打印结果
        print_boxes(final_boxes);
        
        // 打印时间统计
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "时间统计" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  加载点云:    " << std::setw(8) << load_time << " ms" << std::endl;
        std::cout << "  体素化:      " << std::setw(8) << voxel_time << " ms" << std::endl;
        std::cout << "  PFN初始化:   " << std::setw(8) << pfn_init_time << " ms" << std::endl;
        std::cout << "  PFN+Scatter: " << std::setw(8) << pfn_time << " ms" << std::endl;
        std::cout << "  RPN推理:     " << std::setw(8) << rpn_time << " ms" << std::endl;
        std::cout << "  Decode+NMS:  " << std::setw(8) << decode_time << " ms" << std::endl;
        std::cout << "  " << std::string(76, '-') << std::endl;
        std::cout << "  总计:        " << std::setw(8) << total_time << " ms" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ 错误: " << e.what() << std::endl;
        return 1;
    }
}
