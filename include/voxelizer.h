#ifndef VOXELIZER_H
#define VOXELIZER_H

#include <vector>
#include <array>
#include <cstring>

struct VoxelConfig {
    int max_num_points = 32;
    std::array<float, 6> point_cloud_range = {0, -39.68, -3, 69.12, 39.68, 1};
    std::array<float, 3> voxel_size = {0.16, 0.16, 4};
    int max_voxels = 40000;  // for testing
};

struct VoxelData {
    std::vector<float> voxels;           // [num_voxels, max_num_points, 4]
    std::vector<int> coordinates;        // [num_voxels, 4] (batch_id, z, y, x)
    std::vector<int> num_points;         // [num_voxels]
    int num_voxels = 0;
};

class Voxelizer {
public:
    Voxelizer(const VoxelConfig& config);
    
    VoxelData generate(const std::vector<float>& points);
    
private:
    VoxelConfig config_;
    std::array<int, 3> grid_size_;
    
    int point_to_voxel_index(float x, float y, float z);
    std::array<int, 3> point_to_grid_coords(float x, float y, float z);
};

#endif // VOXELIZER_H
