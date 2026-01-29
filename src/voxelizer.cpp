#include "voxelizer.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <unordered_map>

Voxelizer::Voxelizer(const VoxelConfig& config) : config_(config) {
    // Calculate grid size based on point cloud range and voxel size
    grid_size_[0] = static_cast<int>(
        std::ceil((config_.point_cloud_range[3] - config_.point_cloud_range[0]) / config_.voxel_size[0])
    );
    grid_size_[1] = static_cast<int>(
        std::ceil((config_.point_cloud_range[4] - config_.point_cloud_range[1]) / config_.voxel_size[1])
    );
    grid_size_[2] = static_cast<int>(
        std::ceil((config_.point_cloud_range[5] - config_.point_cloud_range[2]) / config_.voxel_size[2])
    );
    
    std::cout << "Voxelizer initialized with grid size: [" 
              << grid_size_[0] << ", " << grid_size_[1] << ", " << grid_size_[2] << "]" << std::endl;
}

std::array<int, 3> Voxelizer::point_to_grid_coords(float x, float y, float z) {
    std::array<int, 3> coords;
    
    // Check if point is within range
    if (x < config_.point_cloud_range[0] || x >= config_.point_cloud_range[3] ||
        y < config_.point_cloud_range[1] || y >= config_.point_cloud_range[4] ||
        z < config_.point_cloud_range[2] || z >= config_.point_cloud_range[5]) {
        return {-1, -1, -1};  // Out of range
    }
    
    coords[0] = static_cast<int>((x - config_.point_cloud_range[0]) / config_.voxel_size[0]);
    coords[1] = static_cast<int>((y - config_.point_cloud_range[1]) / config_.voxel_size[1]);
    coords[2] = static_cast<int>((z - config_.point_cloud_range[2]) / config_.voxel_size[2]);
    
    return coords;
}

VoxelData Voxelizer::generate(const std::vector<float>& points) {
    VoxelData result;
    
    // Assuming points are in format [x, y, z, intensity, ...]
    // We'll use the first 4 values per point
    int num_points = points.size() / 4;
    
    // Map from voxel coordinates to list of point indices
    std::unordered_map<int, std::vector<int>> voxel_map;
    
    // Process each point
    for (int i = 0; i < num_points; ++i) {
        float x = points[i * 4 + 0];
        float y = points[i * 4 + 1];
        float z = points[i * 4 + 2];
        float intensity = points[i * 4 + 3];
        
        auto coords = point_to_grid_coords(x, y, z);
        
        if (coords[0] < 0 || coords[1] < 0 || coords[2] < 0) {
            continue;  // Skip out-of-range points
        }
        
        // Create a unique key for this voxel
        int voxel_key = coords[0] * grid_size_[1] * grid_size_[2] + 
                        coords[1] * grid_size_[2] + 
                        coords[2];
        
        voxel_map[voxel_key].push_back(i);
    }
    
    // Build voxel data
    result.voxels.resize(voxel_map.size() * config_.max_num_points * 4, 0.0f);
    result.coordinates.resize(voxel_map.size() * 4, 0);
    result.num_points.resize(voxel_map.size(), 0);
    result.num_voxels = 0;
    
    int voxel_idx = 0;
    for (const auto& [voxel_key, point_indices] : voxel_map) {
        if (voxel_idx >= config_.max_voxels) {
            break;
        }
        
        // Decode voxel coordinates
        int z_coord = voxel_key % grid_size_[2];
        int y_coord = (voxel_key / grid_size_[2]) % grid_size_[1];
        int x_coord = voxel_key / (grid_size_[1] * grid_size_[2]);
        
        // Store coordinates (batch_id=0, z, y, x)
        result.coordinates[voxel_idx * 4 + 0] = 0;  // batch_id
        result.coordinates[voxel_idx * 4 + 1] = z_coord;
        result.coordinates[voxel_idx * 4 + 2] = y_coord;
        result.coordinates[voxel_idx * 4 + 3] = x_coord;
        
        // Store points in this voxel
        int num_pts = std::min(static_cast<int>(point_indices.size()), config_.max_num_points);
        result.num_points[voxel_idx] = num_pts;
        
        for (int p = 0; p < num_pts; ++p) {
            int point_idx = point_indices[p];
            result.voxels[voxel_idx * config_.max_num_points * 4 + p * 4 + 0] = points[point_idx * 4 + 0];
            result.voxels[voxel_idx * config_.max_num_points * 4 + p * 4 + 1] = points[point_idx * 4 + 1];
            result.voxels[voxel_idx * config_.max_num_points * 4 + p * 4 + 2] = points[point_idx * 4 + 2];
            result.voxels[voxel_idx * config_.max_num_points * 4 + p * 4 + 3] = points[point_idx * 4 + 3];
        }
        
        voxel_idx++;
    }
    
    result.num_voxels = voxel_idx;
    
    // Trim to actual size
    result.voxels.resize(result.num_voxels * config_.max_num_points * 4);
    result.coordinates.resize(result.num_voxels * 4);
    result.num_points.resize(result.num_voxels);
    
    std::cout << "Voxelization complete: " << result.num_voxels << " voxels" << std::endl;
    
    return result;
}
