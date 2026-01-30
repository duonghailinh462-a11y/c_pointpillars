#include "pfn.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

void PFN_CPU::process_voxel(
    const float* voxel_points,
    int num_pts,
    float* output_feature) {
    
    // 从权重大小推断维度
    const size_t output_dim = pfn_bias.size();
    const size_t input_dim = pfn_weights.size() / output_dim;
    
    if (pfn_weights.size() != input_dim * output_dim) {
        throw std::runtime_error("PFN weights size mismatch: expected " + 
                                 std::to_string(input_dim * output_dim) + 
                                 ", got " + std::to_string(pfn_weights.size()));
    }
    if (pfn_bias.size() != output_dim) {
        throw std::runtime_error("PFN bias size mismatch: expected " + 
                                 std::to_string(output_dim) + 
                                 ", got " + std::to_string(pfn_bias.size()));
    }
    
    // 检查输入点特征维度是否匹配
    // 注意：voxel_points 的每个点可能有 input_dim 维，但 Voxelizer 只输出 4 维
    // 如果 input_dim > 4，需要在这里做特征扩展（归一化坐标等）
    
    // 初始化输出为负无穷（用于 max pooling）
    std::fill(output_feature, output_feature + output_dim, -1e9f);
    
    // 对每个点做线性变换：output = max(weight @ point + bias)
    for (int p = 0; p < num_pts; ++p) {
        const float* raw_point = voxel_points + p * 4;  // Voxelizer 输出: [x, y, z, intensity]
        
        // 如果 input_dim > 4，需要扩展特征（例如添加归一化坐标）
        // PointPillars 通常的扩展方式：
        // [x, y, z, intensity, x_norm, y_norm, z_norm, ...]
        std::vector<float> point_feature(input_dim, 0.0f);
        
        if (input_dim >= 4) {
            // 前4维：原始特征
            point_feature[0] = raw_point[0];  // x
            point_feature[1] = raw_point[1];  // y
            point_feature[2] = raw_point[2];  // z
            point_feature[3] = raw_point[3];  // intensity
        }
        
        // 如果 input_dim > 4，添加归一化坐标（相对于 voxel 中心）
        // 这里简化处理：如果 input_dim=10，可能是 [x,y,z,intensity, x_norm,y_norm,z_norm, x^2,y^2,z^2]
        // 但为了简化，我们先用零填充，你可以根据实际训练时的特征工程调整
        // TODO: 根据实际 PFN 输入特征格式调整这里
        
        // GEMM: output = point_feature @ weight^T + bias
        for (size_t o = 0; o < output_dim; ++o) {
            float sum = pfn_bias[o];
            for (size_t i = 0; i < input_dim; ++i) {
                sum += point_feature[i] * pfn_weights[i * output_dim + o];
            }
            // Max pooling across points
            output_feature[o] = std::max(output_feature[o], sum);
        }
    }
}

void PFN_CPU::run(const VoxelInfo& voxel_data, float* rpn_input_map) {
    // RPN 输入尺寸: [1, 64, 496, 432] NCHW
    const int N = 1;
    const int C = 64;
    const int H = 496;
    const int W = 432;
    
    // 清零 BEV map
    std::memset(rpn_input_map, 0, N * C * H * W * sizeof(float));
    
    // 处理每个 voxel
    for (int v = 0; v < voxel_data.num_voxels; ++v) {
        // 获取该 voxel 的坐标 (batch, z, y, x)
        const int* coords = voxel_data.coordinates + v * 4;
        int batch = coords[0];
        // int z = coords[1];  // z 维度在 BEV 中被压缩了，不需要
        int y = coords[2];
        int x = coords[3];
        
        // 检查坐标范围
        if (y < 0 || y >= H || x < 0 || x >= W) {
            continue;
        }
        
        // 处理该 voxel，得到 64 维特征
        float feature[64];
        const float* voxel_pts = voxel_data.voxels + v * voxel_data.max_points * 4;
        int num_pts = voxel_data.num_points[v];
        
        process_voxel(voxel_pts, num_pts, feature);
        
        // Scatter: 直接赋值到 BEV grid (不是 max)
        // NCHW layout: [batch][channel][y][x]
        for (int c = 0; c < C; ++c) {
            int idx = batch * (C * H * W) + c * (H * W) + y * W + x;
            rpn_input_map[idx] = feature[c];
        }
    }
}
