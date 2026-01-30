#pragma once

#include <vector>
#include <cstring>

// PFN 输入：VoxelData（来自 Voxelizer）
// 输出：每个 voxel 的 64 维特征，然后 scatter 到 BEV grid
struct VoxelInfo {
    const float* voxels;        // [num_voxels, max_points, 4]
    const int* coordinates;     // [num_voxels, 4] (batch, z, y, x)
    const int* num_points;       // [num_voxels]
    int num_voxels;
    int max_points;             // 通常是 32
};

class PFN_CPU {
public:
    std::vector<float> pfn_weights;  // 权重矩阵: [input_dim, 64]
    std::vector<float> pfn_bias;     // 偏置: [64]

    // 运行 PFN + Scatter
    // 输入: voxel_data (来自 Voxelizer)
    // 输出: rpn_input_map [1, 64, 496, 432] NCHW，直接写入
    void run(const VoxelInfo& voxel_data, float* rpn_input_map);

private:
    // 单个 voxel 的 PFN 前向：对每个点做线性变换，然后 max pooling
    void process_voxel(
        const float* voxel_points,  // [max_points, 4]
        int num_pts,
        float* output_feature       // [64]
    );
};
