#ifndef LA3DM_BGKL_OCTOMAP_H
#define LA3DM_BGKL_OCTOMAP_H

#include <unordered_map>
#include <vector>
#include <deque>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "rtree.h"
#include "bgklblock.h"
#include "bgkloctree_node.h"
#include "point6f.h"

namespace la3dm {

    /// Single pose history entry: world-frame position + unit quaternion.
    struct PoseEntry {
        float px, py, pz;
        float qx, qy, qz, qw;
    };

    /// PCL PointCloud types as input
    typedef pcl::PointXYZI PCLPointType;
    typedef pcl::PointCloud<PCLPointType> PCLPointCloud;

    /*
     * @brief BGKLOctoMap
     *
     * Bayesian Generalized Kernel Inference for Occupancy Map Prediction
     * The space is partitioned by Blocks in which OcTrees with fixed
     * depth are rooted. Occupancy values in one Block is predicted by 
     * its ExtendedBlock via Bayesian generalized kernel inference.
     */
    // Forward declaration so BGKLPreparedUpdate can hold BGKL3f* without
    // pulling in bgklinference.h (and its Eigen dependency) into this header.
    template<int dim, typename T> class BGKLInference;
    typedef BGKLInference<3, float> BGKL3f;

    /// Intermediate result of the lock-free prepare phase of a map update.
    /// Owns the heap-allocated BGKL3f kernel objects; destructor is defined in
    /// bgkloctomap.cpp where BGKLInference is complete.
    struct BGKLPreparedUpdate {
        std::vector<BlockHashKey>                  test_blocks;
        std::unordered_map<BlockHashKey, BGKL3f*>  bgkl_arr;
        point3f                                     origin;
        point3f                                     sensor_up;
        bool                                        empty = true;
        float                                       w_pose = 1.0f;

        BGKLPreparedUpdate() = default;
        BGKLPreparedUpdate(BGKLPreparedUpdate&&) = default;
        BGKLPreparedUpdate& operator=(BGKLPreparedUpdate&&) = default;
        BGKLPreparedUpdate(const BGKLPreparedUpdate&) = delete;
        BGKLPreparedUpdate& operator=(const BGKLPreparedUpdate&) = delete;

        ~BGKLPreparedUpdate();  // defined in bgkloctomap.cpp
    };

    class BGKLOctoMap {
    public:
        /// Types used internally
        typedef std::vector<point3f> PointCloud;
        typedef std::pair<point3f, float> GPPointType;
        typedef std::pair<point6f, float> GPLineType; // generalizes GPPointType
        typedef std::vector<GPPointType> GPPointCloud;
        typedef std::vector<GPLineType> GPLineCloud; // generalizes GPLineType
        typedef RTree<int, float, 3, float> MyRTree;

    public:
        BGKLOctoMap();

        /*
         * @param resolution (default 0.1m)
         * @param block_depth maximum depth of OcTree (default 4)
         * @param sf2 signal variance in GPs (default 1.0)
         * @param ell length-scale in GPs (default 1.0)
         * @param noise noise variance in GPs (default 0.01)
         * @param l length-scale in logistic regression function (default 100)
         * @param min_var minimum variance in Occupancy (default 0.001)
         * @param max_var maximum variance in Occupancy (default 1000)
         * @param max_known_var maximum variance for Occuapncy to be classified as KNOWN State (default 0.02)
         * @param free_thresh free threshold for Occupancy probability (default 0.3)
         * @param occupied_thresh occupied threshold for Occupancy probability (default 0.7)
         */
        BGKLOctoMap(float resolution,
                unsigned short block_depth,
                float sf2,
                float ell,
                float free_thresh,
                float occupied_thresh,
                float var_thresh,
                float prior_A,
                float prior_B,
                float theta_bw = 0.6f * 3.14159f / 180.0f,
                float phi_bw = 20.0f * 3.14159f / 180.0f,
                bool free_ray_range_weight = false);

        ~BGKLOctoMap();

        /// Set resolution.
        void set_resolution(float resolution);

        /// Set block max depth.
        void set_block_depth(unsigned short max_depth);

        /// Get resolution.
        inline float get_resolution() const { return resolution; }

        /// Get block max depth.
        inline float get_block_depth() const { return block_depth; }

        /*
         * @brief Insert PCL PointCloud into BGKLOctoMaps.
         * @param cloud one scan in PCLPointCloud format
         * @param origin sensor origin in the scan
         * @param sensor_up up vector of the sensor
         * @param ds_resolution downsampling resolution for PCL VoxelGrid filtering (-1 if no downsampling)
         * @param free_res resolution for sampling free training points along sensor beams (default 2.0)
         * @param max_range maximum range for beams to be considered as valid measurements (-1 if no limitation)
         */
        void insert_pointcloud(const PCLPointCloud &cloud, const point3f &origin,
                               const point3f &sensor_up,
                               float ds_resolution,
                               float free_res = 2.0f,
                               float max_range = -1,
                               float qx = 0.f, float qy = 0.f, float qz = 0.f, float qw = 1.f);

        /// Lock-free phase: downsampling, ray tracing, BGKL kernel training.
        /// Does NOT access block_arr — safe to call without holding ot_mutex_.
        BGKLPreparedUpdate prepare_pointcloud_update(const PCLPointCloud &cloud,
                                                     const point3f &origin,
                                                     const point3f &sensor_up,
                                                     float ds_resolution,
                                                     float free_res,
                                                     float max_range,
                                                     float qx = 0.f, float qy = 0.f,
                                                     float qz = 0.f, float qw = 1.f);

        /// Voxel-level change record produced by commit_pointcloud_update().
        /// Callers accumulate these into a dirty buffer so that computePriorityCache()
        /// can update the candidate index incrementally instead of re-scanning all leaves.
        struct DirtyEntry {
            point3f      pos;        // world-frame voxel centre
            float        priority;   // node.get_var() after update
            float        info[6];    // raw symmetric info matrix [Ixx,Ixy,Ixz,Iyy,Iyz,Izz]
            bool         active;     // has_active_info_matrix() && priority > 1e-8f
            la3dm::State state;      // OCCUPIED / FREE / UNKNOWN
        };

        /// Write phase: prediction loop + node.update(). Must be called under unique_lock(ot_mutex_).
        /// If dirty_out is non-null, a DirtyEntry is appended for every voxel whose info
        /// matrix was updated, allowing callers to maintain an incremental candidate index.
        void commit_pointcloud_update(const BGKLPreparedUpdate &upd,
                                      std::vector<DirtyEntry>* dirty_out = nullptr);

        /// Configure pose-level novelty weighting. Call once after construction.
        /// When enabled, the per-frame w_pose replaces the per-voxel w_novelty in the
        /// Beta update path. The info matrix update is unchanged.
        void configure_pose_level_weighting(bool enabled, int K, float sigma,
                                            float w_roll, float w_pitch, float w_yaw,
                                            float w_vx_l2, float w_vy_l2, float w_vz_l2);

        /// Configure frustum-based block culling. Call once after construction.
        /// @param swath_angle_deg  Total multibeam swath width in degrees
        ///                         (e.g. 120 degrees). Pass <= 0 to disable azimuth
        ///                         culling (elevation culling via phi_bw is always on).
        void configure_frustum(float swath_angle_deg);

        void insert_training_data(const GPLineCloud &cloud);

        /// Get bounding box of the map.
        void get_bbox(point3f &lim_min, point3f &lim_max) const;

        class RayCaster {
        public:
            RayCaster(const BGKLOctoMap *map, const point3f &start, const point3f &end) : map(map) {
                assert(map != nullptr);

                _block_key = block_to_hash_key(start);
                block = map->search(_block_key);
                lim = static_cast<unsigned short>(pow(2, map->block_depth - 1));

                if (block != nullptr) {
                    block->get_index(start, x, y, z);
                    block_lim = block->get_center();
                    block_size = block->size;
                    current_p = start;
                    resolution = map->resolution;

                    int x0 = static_cast<int>((start.x() / resolution));
                    int y0 = static_cast<int>((start.y() / resolution));
                    int z0 = static_cast<int>((start.z() / resolution));
                    int x1 = static_cast<int>((end.x() / resolution));
                    int y1 = static_cast<int>((end.y() / resolution));
                    int z1 = static_cast<int>((end.z() / resolution));
                    dx = abs(x1 - x0);
                    dy = abs(y1 - y0);
                    dz = abs(z1 - z0);
                    n = 1 + dx + dy + dz;
                    x_inc = x1 > x0 ? 1 : (x1 == x0 ? 0 : -1);
                    y_inc = y1 > y0 ? 1 : (y1 == y0 ? 0 : -1);
                    z_inc = z1 > z0 ? 1 : (z1 == z0 ? 0 : -1);
                    xy_error = dx - dy;
                    xz_error = dx - dz;
                    yz_error = dy - dz;
                    dx *= 2;
                    dy *= 2;
                    dz *= 2;
                } else {
                    n = 0;
                }
            }

            inline bool end() const { return n <= 0; }

            bool next(point3f &p, OcTreeNode &node, BlockHashKey &block_key, OcTreeHashKey &node_key) {
                assert(!end());
                bool valid = false;
                unsigned short index = x + y * lim + z * lim * lim;
                node_key = Block::index_map[index];
                block_key = _block_key;
                if (block != nullptr) {
                    valid = true;
                    node = (*block)[node_key];
                    current_p = block->get_point(x, y, z);
                    p = current_p;
                } else {
                    p = current_p;
                }

                if (xy_error > 0 && xz_error > 0) {
                    x += x_inc;
                    current_p.x() += x_inc * resolution;
                    xy_error -= dy;
                    xz_error -= dz;
                    if (x >= lim || x < 0) {
                        block_lim.x() += x_inc * block_size;
                        _block_key = block_to_hash_key(block_lim);
                        block = map->search(_block_key);
                        x = x_inc > 0 ? 0 : lim - 1;
                    }
                } else if (xy_error < 0 && yz_error > 0) {
                    y += y_inc;
                    current_p.y() += y_inc * resolution;
                    xy_error += dx;
                    yz_error -= dz;
                    if (y >= lim || y < 0) {
                        block_lim.y() += y_inc * block_size;
                        _block_key = block_to_hash_key(block_lim);
                        block = map->search(_block_key);
                        y = y_inc > 0 ? 0 : lim - 1;
                    }
                } else if (yz_error < 0 && xz_error < 0) {
                    z += z_inc;
                    current_p.z() += z_inc * resolution;
                    xz_error += dx;
                    yz_error += dy;
                    if (z >= lim || z < 0) {
                        block_lim.z() += z_inc * block_size;
                        _block_key = block_to_hash_key(block_lim);
                        block = map->search(_block_key);
                        z = z_inc > 0 ? 0 : lim - 1;
                    }
                } else if (xy_error == 0) {
                    x += x_inc;
                    y += y_inc;
                    n -= 2;
                    current_p.x() += x_inc * resolution;
                    current_p.y() += y_inc * resolution;
                    if (x >= lim || x < 0) {
                        block_lim.x() += x_inc * block_size;
                        _block_key = block_to_hash_key(block_lim);
                        block = map->search(_block_key);
                        x = x_inc > 0 ? 0 : lim - 1;
                    }
                    if (y >= lim || y < 0) {
                        block_lim.y() += y_inc * block_size;
                        _block_key = block_to_hash_key(block_lim);
                        block = map->search(_block_key);
                        y = y_inc > 0 ? 0 : lim - 1;
                    }
                }
                n--;
                return valid;
            }

        private:
            const BGKLOctoMap *map;
            Block *block;
            point3f block_lim;
            float block_size, resolution;
            int dx, dy, dz, error, n;
            int x_inc, y_inc, z_inc, xy_error, xz_error, yz_error;
            unsigned short index, x, y, z, lim;
            BlockHashKey _block_key;
            point3f current_p;
        };

        /// LeafIterator for iterating all leaf nodes in blocks
        class LeafIterator : public std::iterator<std::forward_iterator_tag, OcTreeNode> {
        public:
            LeafIterator(const BGKLOctoMap *map) {
                assert(map != nullptr);

                block_it = map->block_arr.cbegin();
                end_block = map->block_arr.cend();

                if (map->block_arr.size() > 0) {
                    leaf_it = block_it->second->begin_leaf();
                    end_leaf = block_it->second->end_leaf();
                } else {
                    leaf_it = OcTree::LeafIterator();
                    end_leaf = OcTree::LeafIterator();
                }
            }

            // Spatial-filter constructor: only walks leaves of blocks whose
            // center is within `radius + block_half_diagonal` of `center`.
            // Uses grid enumeration (O(r³/block_size³)) instead of a linear
            // scan of all blocks (O(N_total_blocks)) — stays fast as the map grows.
            LeafIterator(const BGKLOctoMap *map, const point3f &center, float radius)
                    : filter_active(true), filter_center(center) {
                assert(map != nullptr);
                float bs      = map->block_size;
                float padded  = radius + 0.866025404f * bs;
                block_threshold_sq = padded * padded;
                end_block = map->block_arr.cend();

                // Enumerate all block grid cells whose center is within the
                // padded sphere and perform direct hash-table lookups.
                int ix_min = (int)std::floor((center.x() - padded) / bs + 0.5f);
                int ix_max = (int)std::floor((center.x() + padded) / bs + 0.5f);
                int iy_min = (int)std::floor((center.y() - padded) / bs + 0.5f);
                int iy_max = (int)std::floor((center.y() + padded) / bs + 0.5f);
                int iz_min = (int)std::floor((center.z() - padded) / bs + 0.5f);
                int iz_max = (int)std::floor((center.z() + padded) / bs + 0.5f);

                for (int ix = ix_min; ix <= ix_max; ++ix) {
                    float bx = ix * bs;
                    float dx = bx - center.x();
                    for (int iy = iy_min; iy <= iy_max; ++iy) {
                        float by = iy * bs;
                        float dy = by - center.y();
                        for (int iz = iz_min; iz <= iz_max; ++iz) {
                            float bz = iz * bs;
                            float dz = bz - center.z();
                            if (dx*dx + dy*dy + dz*dz > block_threshold_sq) continue;
                            BlockHashKey key = block_to_hash_key(bx, by, bz);
                            auto it = map->block_arr.find(key);
                            if (it == map->block_arr.end()) continue;
                            sphere_block_iters_.push_back(it);
                        }
                    }
                }

                sphere_idx_ = 0;
                if (!sphere_block_iters_.empty()) {
                    block_it = sphere_block_iters_[0];
                    leaf_it  = block_it->second->begin_leaf();
                    end_leaf = block_it->second->end_leaf();
                } else {
                    block_it = end_block;
                    leaf_it  = OcTree::LeafIterator();
                    end_leaf = OcTree::LeafIterator();
                }
            }

            // just for initializing end iterator
            LeafIterator(std::unordered_map<BlockHashKey, Block *>::const_iterator block_it,
                         OcTree::LeafIterator leaf_it)
                    : block_it(block_it), leaf_it(leaf_it), end_block(block_it), end_leaf(leaf_it) { }

            bool operator==(const LeafIterator &other) {
                return (block_it == other.block_it) && (leaf_it == other.leaf_it);
            }

            bool operator!=(const LeafIterator &other) {
                return !(this->operator==(other));
            }

            LeafIterator operator++(int) {
                LeafIterator result(*this);
                ++(*this);
                return result;
            }

            LeafIterator &operator++() {
                ++leaf_it;
                if (leaf_it == end_leaf) {
                    if (filter_active) {
                        ++sphere_idx_;
                        if (sphere_idx_ < sphere_block_iters_.size()) {
                            block_it = sphere_block_iters_[sphere_idx_];
                            leaf_it  = block_it->second->begin_leaf();
                            end_leaf = block_it->second->end_leaf();
                        } else {
                            block_it = end_block;
                            leaf_it  = OcTree::LeafIterator();
                            end_leaf = OcTree::LeafIterator();
                        }
                    } else {
                        ++block_it;
                        if (block_it != end_block) {
                            leaf_it = block_it->second->begin_leaf();
                            end_leaf = block_it->second->end_leaf();
                        }
                    }
                }
                return *this;
            }

            OcTreeNode &operator*() const {
                return *leaf_it;
            }

            std::vector<point3f> get_pruned_locs() const {
                std::vector<point3f> pruned_locs;
                point3f center = get_loc();
                float size = get_size();
                float x0 = center.x() - size * 0.5 + Block::resolution * 0.5;
                float y0 = center.y() - size * 0.5 + Block::resolution * 0.5;
                float z0 = center.z() - size * 0.5 + Block::resolution * 0.5;
                float x1 = center.x() + size * 0.5;
                float y1 = center.y() + size * 0.5;
                float z1 = center.z() + size * 0.5;
                for (float x = x0; x < x1; x += Block::resolution) {
                    for (float y = y0; y < y1; y += Block::resolution) {
                        for (float z = z0; z < z1; z += Block::resolution) {
                            pruned_locs.emplace_back(x, y, z);
                        }
                    }
                }
                return pruned_locs;
            }

            inline OcTreeNode &get_node() const {
                return operator*();
            }

            inline point3f get_loc() const {
                return block_it->second->get_loc(leaf_it);
            }

            inline float get_size() const {
                return block_it->second->get_size(leaf_it);
            }

        private:
            std::unordered_map<BlockHashKey, Block *>::const_iterator block_it;
            std::unordered_map<BlockHashKey, Block *>::const_iterator end_block;

            OcTree::LeafIterator leaf_it;
            OcTree::LeafIterator end_leaf;

            // Optional spatial filter (active only when constructed with a
            // center+radius). Default-initialised so the other constructors
            // leave filtering disabled.
            bool filter_active = false;
            point3f filter_center;
            float block_threshold_sq = 0.0f;

            // Pre-built list of matching blocks for the sphere constructor.
            // Avoids the O(N_total_blocks) linear scan of advance_past_filtered_blocks.
            std::vector<std::unordered_map<BlockHashKey, Block *>::const_iterator> sphere_block_iters_;
            std::size_t sphere_idx_ = 0;
        };

        /// @return the beginning of leaf iterator
        inline LeafIterator begin_leaf() const { return LeafIterator(this); }

        /// @return a leaf iterator restricted to blocks whose center is within
        ///         `radius + block_half_diagonal` of `center`. Use with the
        ///         unchanged `end_leaf()` sentinel. Falls back to full walk if
        ///         you pass a radius large enough to enclose the map.
        inline LeafIterator begin_leaf_in_sphere(const point3f &center, float radius) const {
            return LeafIterator(this, center, radius);
        }

        /// @return the end of leaf iterator
        inline LeafIterator end_leaf() const { return LeafIterator(block_arr.cend(), OcTree::LeafIterator()); }

        OcTreeNode search(point3f p) const;

        OcTreeNode search(float x, float y, float z) const;

        Block *search(BlockHashKey key) const;

        inline float get_block_size() const { return block_size; }

    private:
        /// @return true if point is inside a bounding box given min and max limits.
        inline bool gp_point_in_bbox(const GPPointType &p, const point3f &lim_min, const point3f &lim_max) const {
            return (p.first.x() > lim_min.x() && p.first.x() < lim_max.x() &&
                    p.first.y() > lim_min.y() && p.first.y() < lim_max.y() &&
                    p.first.z() > lim_min.z() && p.first.z() < lim_max.z());
        }

        /// Get the bounding box of a pointcloud.
        void bbox(const GPLineCloud &cloud, point3f &lim_min, point3f &lim_max) const;

        /// Get all block indices inside a bounding box, with optional frustum culling.
        void get_blocks_in_bbox(const point3f &lim_min, const point3f &lim_max,
                                std::vector<BlockHashKey> &blocks,
                                const point3f &origin,
                                const point3f &sensor_up,
                                const point3f &forward_hat) const;

        /// Get all points inside a bounding box assuming pointcloud has been inserted in rtree before.
        int get_gp_points_in_bbox(const point3f &lim_min, const point3f &lim_max,
                                  std::vector<int> &out);

        /// @return true if point exists inside a bounding box assuming pointcloud has been inserted in rtree before.
        int has_gp_points_in_bbox(const point3f &lim_min, const point3f &lim_max);

        /// Get all points inside a bounding box (block) assuming pointcloud has been inserted in rtree before.
        int get_gp_points_in_bbox(const BlockHashKey &key, std::vector<int> &out);

        /// @return true if point exists inside a bounding box (block) assuming pointcloud has been inserted in rtree before.
        int has_gp_points_in_bbox(const BlockHashKey &key);

        /// Get all points inside an extended block assuming pointcloud has been inserted in rtree before.
        int get_gp_points_in_bbox(const ExtendedBlock &block, std::vector<int> &out);

        /// @return true if point exists inside an extended block assuming pointcloud has been inserted in rtree before.
        int has_gp_points_in_bbox(const ExtendedBlock &block);

        /// RTree callback function
        static bool count_callback(int p, void *arg);

        /// RTree callback function
        static bool search_callback(int p, void *arg);

        /// Downsample PCLPointCloud using PCL VoxelGrid Filtering.
        void downsample(const PCLPointCloud &in, PCLPointCloud &out, float ds_resolution) const;

        /// Sample free training points along sensor beams.
        void beam_sample(const point3f &hits, const point3f &origin, PointCloud &frees,
                         float free_resolution) const;

        /// Get training data from one sensor scan.
        void get_training_data(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                               float free_resolution, float max_range, GPLineCloud &xy, GPLineCloud &rays, std::vector<int> &ray_idx) const;

        float resolution;
        float block_size;
        unsigned short block_depth;
        float theta_bw;
        float phi_bw;
        bool free_ray_range_weight;
        std::unordered_map<BlockHashKey, Block *> block_arr;
        MyRTree rtree;

        // ---- Pose-level novelty weighting ----
        bool pose_level_weighting_ = false;
        std::deque<PoseEntry> pose_history_;
        int   pose_history_K_     = 20;
        float pose_novelty_sigma_ = 0.3f;
        // W diagonal: [w_roll, w_pitch, w_yaw, w_vx/l², w_vy/l², w_vz/l²]
        float pose_w_[6] = {1.0f, 0.6f, 0.05f, 0.2f, 0.05f, 0.5f};

        // ---- Frustum culling ----
        // swath_half_angle_ is half the total multibeam swath angle in radians.
        // Defaults to M_PI (no azimuth culling). Elevation culling uses phi_bw always.
        float swath_half_angle_ = (float)M_PI;

        float compute_w_pose_(float px, float py, float pz,
                              float qx, float qy, float qz, float qw) const;
        void  update_pose_history_(float px, float py, float pz,
                                   float qx, float qy, float qz, float qw);
    };

}

#endif // LA3DM_BGKLOCTOMAP_H
