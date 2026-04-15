#include <algorithm>
#include <ros/ros.h>
#include <pcl/filters/voxel_grid.h>
#include "bgkloctomap.h"
#include "bgklinference.h"

using std::vector;

// #define DEBUG true;

#ifdef DEBUG

#include <iostream>

#define Debug_Msg(msg) {\
std::cout << "Debug: " << msg << std::endl; }
#endif

namespace la3dm {

    BGKLOctoMap::BGKLOctoMap() : BGKLOctoMap(0.1f, // resolution
                                        4, // block_depth
                                        1.0, // sf2
                                        1.0, // ell
                                        0.3f, // free_thresh
                                        0.7f, // occupied_thresh
                                        1.0f, // var_thresh
                                        1.0f, // prior_A
                                        1.0f // prior_B
                                    ) { }

    BGKLOctoMap::BGKLOctoMap(float resolution,
                        unsigned short block_depth,
                        float sf2,
                        float ell,
                        float free_thresh,
                        float occupied_thresh,
                        float var_thresh,
                        float prior_A,
                        float prior_B,
                        float theta_bw,
                        float phi_bw)
            : resolution(resolution), block_depth(block_depth),
              block_size((float) pow(2, block_depth - 1) * resolution),
              theta_bw(theta_bw), phi_bw(phi_bw) {
        Block::resolution = resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
        Block::index_map = init_index_map(Block::key_loc_map, block_depth);

        OcTree::max_depth = block_depth;

        OcTreeNode::sf2 = sf2;
        OcTreeNode::ell = ell;
        OcTreeNode::free_thresh = free_thresh;
        OcTreeNode::occupied_thresh = occupied_thresh;
        OcTreeNode::var_thresh = var_thresh;
        OcTreeNode::prior_A = prior_A;
        OcTreeNode::prior_B = prior_B;
    }

    BGKLOctoMap::~BGKLOctoMap() {
        for (auto it = block_arr.begin(); it != block_arr.end(); ++it) {
            if (it->second != nullptr) {
                delete it->second;
            }
        }
    }

    void BGKLOctoMap::set_resolution(float resolution) {
        this->resolution = resolution;
        Block::resolution = resolution;
        this->block_size = (float) pow(2, block_depth - 1) * resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
    }

    void BGKLOctoMap::set_block_depth(unsigned short max_depth) {
        this->block_depth = max_depth;
        OcTree::max_depth = max_depth;
        this->block_size = (float) pow(2, block_depth - 1) * resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
    }

    void BGKLOctoMap::insert_pointcloud(const PCLPointCloud &cloud, const point3f &origin,
                                      const point3f &sensor_up,
                                      float ds_resolution,
                                      float free_res, float max_range, float w_novelty) {

#ifdef DEBUG
        Debug_Msg("Insert pointcloud: " << "cloud size: " << cloud.size() << " origin: " << origin);
#endif

        ////////// Preparation //////////////////////////
        /////////////////////////////////////////////////
        GPLineCloud xy;
        GPLineCloud rays;
        vector<int> ray_idx;
        // const int ray_size = rays.size();
        // std::array<int, ray_size> ray_keys;
        get_training_data(cloud, origin, ds_resolution, free_res, max_range, xy, rays, ray_idx);
        // vector<int> ray_keys(rays.size(), 0);
        assert (ray_idx.size() == xy.size());
        // std::cout << "N rays: " << rays.size() << std::endl;
        // std::cout << "vec size: " << ray_keys.size() << std::endl;
#ifdef DEBUG
        Debug_Msg("Training data size: " << xy.size());
#endif

        // Prevent Segfault if training data is completely empty
        if (xy.empty()) {
            #ifdef DEBUG
            Debug_Msg("Training data empty after filtering. Skipping map update.");
            #endif
            // ROS_INFO_STREAM("Training data empty after filtering. Skipping map update!!!!!!!!!!!!!!!!!");
            return; 
        }

        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);

        vector<BlockHashKey> blocks;
        get_blocks_in_bbox(lim_min, lim_max, blocks);

        // std::unordered_map<BlockHashKey, GPLineCloud> key_train_data_map;
        for (int k = 0; k < xy.size(); ++k) {
            float p[] = {xy[k].first.x0(), xy[k].first.y0(), xy[k].first.z0()};
            rtree.Insert(p, p, k);

        }
        /////////////////////////////////////////////////

        ////////// Training /////////////////////////////
        /////////////////////////////////////////////////
        vector<BlockHashKey> test_blocks;
        std::unordered_map<BlockHashKey, BGKL3f *> bgkl_arr;

#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < blocks.size(); ++i) {
            BlockHashKey key = blocks[i];
            ExtendedBlock eblock = get_extended_block(key);
            if (has_gp_points_in_bbox(eblock))
#ifdef OPENMP
#pragma omp critical
#endif
            {
                test_blocks.push_back(key);
            };

            // GPLineCloud block_xy;
            vector<int> xy_idx;
            get_gp_points_in_bbox(key, xy_idx);
            if (xy_idx.size() < 1)
                continue;

            vector<int> ray_keys(rays.size(), 0);
            vector<float> block_x, block_y, block_w;
            for (int j = 0; j < xy_idx.size(); ++j) {
#ifdef OPENMP
#pragma omp critical
#endif
            {
                if (ray_idx[xy_idx[j]] == -1) {
                    block_x.push_back(xy[xy_idx[j]].first.x0());
                    block_x.push_back(xy[xy_idx[j]].first.y0());
                    block_x.push_back(xy[xy_idx[j]].first.z0());
                    block_x.push_back(xy[xy_idx[j]].first.x0());
                    block_x.push_back(xy[xy_idx[j]].first.y0());
                    block_x.push_back(xy[xy_idx[j]].first.z0());
                    block_y.push_back(1.0f);
                    block_w.push_back(xy[xy_idx[j]].second); // weight computed based on range
                }
                else if (ray_keys[ray_idx[xy_idx[j]]] == 0) {
                    ray_keys[ray_idx[xy_idx[j]]] = 1;
                    block_x.push_back(rays[ray_idx[xy_idx[j]]].first.x0());
                    block_x.push_back(rays[ray_idx[xy_idx[j]]].first.y0());
                    block_x.push_back(rays[ray_idx[xy_idx[j]]].first.z0());
                    block_x.push_back(rays[ray_idx[xy_idx[j]]].first.x1());
                    block_x.push_back(rays[ray_idx[xy_idx[j]]].first.y1());
                    block_x.push_back(rays[ray_idx[xy_idx[j]]].first.z1());
                    block_y.push_back(0.0f);
                    block_w.push_back(1.0f); // free space ray gets full evidence weight
                }
            }
            };
            // std::cout << "number of training blocks" << block_y.size() << std::endl;
            BGKL3f *bgkl = new BGKL3f(OcTreeNode::sf2, OcTreeNode::ell, theta_bw, phi_bw, w_novelty);
            bgkl->train(block_x, block_y, block_w);
#ifdef OPENMP
#pragma omp critical
#endif
            {
                bgkl_arr.emplace(key, bgkl);
            };
        }
#ifdef DEBUG
        Debug_Msg("Training done");
        Debug_Msg("Prediction: block number: " << test_blocks.size());
#endif
        /////////////////////////////////////////////////

        ////////// Prediction ///////////////////////////
        /////////////////////////////////////////////////
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < test_blocks.size(); ++i) {
            BlockHashKey key = test_blocks[i];
#ifdef OPENMP
#pragma omp critical
#endif
            {
                if (block_arr.find(key) == block_arr.end())
                    block_arr.emplace(key, new Block(hash_key_to_block(key)));
            };
            Block *block = block_arr[key];
            vector<float> xs;
            for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it) {
                point3f p = block->get_loc(leaf_it);
                xs.push_back(p.x());
                xs.push_back(p.y());
                xs.push_back(p.z());
            }

            ExtendedBlock eblock = block->get_extended_block();
            for (auto block_it = eblock.cbegin(); block_it != eblock.cend(); ++block_it) {
                auto bgkl = bgkl_arr.find(*block_it);
                if (bgkl == bgkl_arr.end())
                    continue;

                vector<float> ybar, kbar;
                bgkl->second->predict(origin, sensor_up, xs, ybar, kbar);

                int j = 0;
                for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it, ++j) {
                    OcTreeNode &node = leaf_it.get_node();
                    auto node_loc = block->get_loc(leaf_it);

                    float obs_range = (float)(node_loc - origin).norm();

                    // Only need to update if kernel density total kernel density est > 0
                    if (kbar[j] > 0.000001f) {
                        // ---- Compute constraint direction n ----
                        // n = normalize(sensor_up - (sensor_up · loŝ) · loŝ)
                        // where loŝ = normalize(node_loc - origin)
                        // sensor_up is the sonar's vertical axis (beam-spread direction) in world frame
                        point3f los_vec = node_loc - origin;
                        point3f n_vec;
                        bool valid_n = false;

                        if (obs_range > 1e-6f) {
                            point3f los_hat = los_vec * (1.0f / obs_range);
                            float dot_su = (float)sensor_up.dot(los_hat);
                            n_vec = sensor_up - los_hat * dot_su;
                            float n_norm = (float)n_vec.norm();
                            if (n_norm > 1e-8f) {
                                n_vec *= (1.0f / n_norm);
                                valid_n = true;
                            }
                        }

                        if (valid_n) {
                            // ---- Per-voxel evidence weight ----
                            // w_voxel = 1 - (nᵀ I n) / (λ_max_cache + ε)
                            float w_voxel = node.compute_w_voxel(n_vec);

                            // ---- Rank-1 information matrix update ----
                            // I(p) += w_range · n nᵀ  (uses w_range only, NOT w_voxel)
                            // Matches training-data formula: w_range ≈ 1/r², floored at 0.05
                            float w_range = 1.0f / (obs_range * obs_range + 1.0f);
                            if (w_range < 0.05f) w_range = 0.05f;
                            node.update_info_matrix(n_vec, w_range);

                            // ---- Beta update with w_total = w_range_baked * w_voxel ----
                            // ybar/kbar already encode w_range (training-data peak_weight).
                            // Scaling by w_voxel gives the combined novelty-weighted evidence.
                            if (w_voxel > 1e-8f)
                                node.update(ybar[j] * w_voxel, kbar[j] * w_voxel, obs_range);

                            // ---- Lifecycle management ----
                            node.check_deallocation();
                        } else {
                            // Degenerate geometry: LOS parallel to sensor_up (rare).
                            // Fall back to full-weight update.
                            node.update(ybar[j], kbar[j], obs_range);
                        }
                    }
                }
            }
        }
#ifdef DEBUG
        Debug_Msg("Prediction done");
#endif
        /////////////////////////////////////////////////

        ////////// Pruning //////////////////////////////
        ///////////////////////////////////////////////
// #ifdef OPENMP
// #pragma omp parallel for
// #endif
        // for (int i = 0; i < test_blocks.size(); ++i) {
        //     BlockHashKey key = test_blocks[i];
        //     auto block = block_arr.find(key);
        //     if (block == block_arr.end())
        //         continue;
        //     block->second->prune();
        // }
// #ifdef DEBUG
//         Debug_Msg("Pruning done");
// #endif
        /////////////////////////////////////////////////


        ////////// Cleaning /////////////////////////////
        /////////////////////////////////////////////////
        for (auto it = bgkl_arr.begin(); it != bgkl_arr.end(); ++it) {
            delete it->second;
        }
        // ray_keys.clear();


        rtree.RemoveAll();
    }

    void BGKLOctoMap::get_bbox(point3f &lim_min, point3f &lim_max) const {
        lim_min = point3f(0, 0, 0);
        lim_max = point3f(0, 0, 0);

        GPLineCloud centers;
        for (auto it = block_arr.cbegin(); it != block_arr.cend(); ++it) {
            centers.emplace_back(point6f(it->second->get_center()), 1);
        }
        if (centers.size() > 0) {
            bbox(centers, lim_min, lim_max);
            lim_min -= point3f(block_size, block_size, block_size) * 0.5;
            lim_max += point3f(block_size, block_size, block_size) * 0.5;
        }
    }

    void BGKLOctoMap::get_training_data(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                                      float free_resolution, float max_range, GPLineCloud &xy, GPLineCloud &rays, vector<int> &ray_idx) const {
        PCLPointCloud sampled_hits;
        downsample(cloud, sampled_hits, ds_resolution);

        // std::cout << "Sampled points: " << sampled_hits.size() << std::endl;

        PCLPointCloud frees;
        frees.height = 1;
        frees.width = 0;
        rays.clear();
        ray_idx.clear();
        xy.clear();
        int idx = 0;
        for (auto it = sampled_hits.begin(); it != sampled_hits.end(); ++it) {
            point3f p(it->x, it->y, it->z);
            // if (max_range > 0) {
            //     double l = (p - origin).norm();
            //     if (l > max_range)
            //         continue;
            // }

            // point6f p6f(p);
            // xy.emplace_back(p6f, 1.0f);
            // ray_idx.push_back(-1);

            float true_dist = (float) sqrt((p.x() - origin.x()) * (p.x() - origin.x()) +
                                           (p.y() - origin.y()) * (p.y() - origin.y()) +
                                           (p.z() - origin.z()) * (p.z() - origin.z()));

            bool is_sentinel = false;
            float l = true_dist;
            
            if (max_range > 0 && true_dist >= max_range) {
                is_sentinel = true;
                l = max_range; // Clamp the ray length so it doesn't clear space to infinity
            }

            bool is_first_hit = (it->intensity > 0.01f);

            float nx = (p.x() - origin.x()) / true_dist;
            float ny = (p.y() - origin.y()) / true_dist;
            float nz = (p.z() - origin.z()) / true_dist;

            point3f occ_endpt(origin.x() + nx * l, origin.y() + ny * l, origin.z() + nz * l);

            if (!is_sentinel) {
                // Hits closer than this get maximum evidence.
                float baseline_range = 1.0f;
                
                // Scale peak weight by observation range
                float peak_weight = baseline_range / (true_dist * true_dist);

                if (peak_weight < 0.05f) peak_weight = 0.05f; // minimum floor
                
                xy.emplace_back(point6f(occ_endpt), peak_weight);
                ray_idx.push_back(-1); // -1 tells the trainer this is an occupied point, not a ray
            }

            // point3f free_endpt(origin.x() + nx * (l - free_resolution), origin.y() + ny * (l - free_resolution), origin.z() + nz * (l - 0.1f));
            // point6f line6f(origin, free_endpt);
            // rays.emplace_back(line6f, 0.0f);

            if (is_first_hit || is_sentinel) {
                float l_free = is_sentinel ? l : (l - free_resolution);

                if (l_free > 0) {
                    int current_ray_idx = rays.size();

                    PointCloud frees_n;
                    beam_sample(occ_endpt, origin, frees_n, free_resolution);

                    // Add origin to xy
                    xy.emplace_back(point6f(origin.x(), origin.y(), origin.z()), 0.0f);
                    ray_idx.push_back(current_ray_idx);

                    // Add sampled points to xy
                    for (auto p_free = frees_n.begin(); p_free != frees_n.end(); ++p_free) {
                        xy.emplace_back(point6f(p_free->x(), p_free->y(), p_free->z()), 0.0f);
                        ray_idx.push_back(current_ray_idx);
                    }

                    point3f free_endpt(origin.x() + nx * l_free, origin.y() + ny * l_free, origin.z() + nz * l_free);
                    point6f line6f(origin, free_endpt);
                    rays.emplace_back(line6f, 0.0f);
                }
            }
        }
    }

    void BGKLOctoMap::downsample(const PCLPointCloud &in, PCLPointCloud &out, float ds_resolution) const {
        if (ds_resolution < 0) {
            out = in;
            return;
        }

        PCLPointCloud::Ptr pcl_in(new PCLPointCloud(in));

        pcl::VoxelGrid<PCLPointType> sor;
        sor.setInputCloud(pcl_in);
        sor.setLeafSize(ds_resolution, ds_resolution, ds_resolution);
        sor.filter(out);
    }

    void BGKLOctoMap::beam_sample(const point3f &hit, const point3f &origin, PointCloud &frees,
                                float free_resolution) const {
        frees.clear();

        float x0 = origin.x();
        float y0 = origin.y();
        float z0 = origin.z();

        float x = hit.x();
        float y = hit.y();
        float z = hit.z();

        float l = (float) sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0));

        float nx = (x - x0) / l;
        float ny = (y - y0) / l;
        float nz = (z - z0) / l;

        float d = l - free_resolution;
        while (d > 0.0) {
            frees.emplace_back(x0 + nx * d, y0 + ny * d, z0 + nz * d);
            d -= free_resolution;
        }
    }

    void BGKLOctoMap::bbox(const GPLineCloud &cloud, point3f &lim_min, point3f &lim_max) const {
        vector<float> x, y, z;
        for (auto it = cloud.cbegin(); it != cloud.cend(); ++it) {
            x.push_back(it->first.x0());
            x.push_back(it->first.x1());
            y.push_back(it->first.y0());
            y.push_back(it->first.y1());
            z.push_back(it->first.z0());
            z.push_back(it->first.z1());
        }

        auto xlim = std::minmax_element(x.cbegin(), x.cend());
        auto ylim = std::minmax_element(y.cbegin(), y.cend());
        auto zlim = std::minmax_element(z.cbegin(), z.cend());

        lim_min.x() = *xlim.first;
        lim_min.y() = *ylim.first;
        lim_min.z() = *zlim.first;

        lim_max.x() = *xlim.second;
        lim_max.y() = *ylim.second;
        lim_max.z() = *zlim.second;
    }

    void BGKLOctoMap::get_blocks_in_bbox(const point3f &lim_min, const point3f &lim_max,
                                   vector<BlockHashKey> &blocks) const {
        // Symmetrical 2-block padding to support wide vertical apertures
        float pad = 2.0f * block_size; 
        for (float x = lim_min.x() - pad; x <= lim_max.x() + pad; x += block_size) {
            for (float y = lim_min.y() - pad; y <= lim_max.y() + pad; y += block_size) {
                for (float z = lim_min.z() - pad; z <= lim_max.z() + pad; z += block_size) {
                    blocks.push_back(block_to_hash_key(x, y, z));
                }
            }
        }
    }

    int BGKLOctoMap::get_gp_points_in_bbox(const BlockHashKey &key,
                                         vector<int> &out) {
        point3f half_size(block_size / 2.0f, block_size / 2.0f, block_size / 2.0);
        point3f lim_min = hash_key_to_block(key) - half_size;
        point3f lim_max = hash_key_to_block(key) + half_size;
        return get_gp_points_in_bbox(lim_min, lim_max, out);
    }

    int BGKLOctoMap::has_gp_points_in_bbox(const BlockHashKey &key) {
        point3f half_size(block_size / 2.0f, block_size / 2.0f, block_size / 2.0);
        point3f lim_min = hash_key_to_block(key) - half_size;
        point3f lim_max = hash_key_to_block(key) + half_size;
        return has_gp_points_in_bbox(lim_min, lim_max);
    }

    int BGKLOctoMap::get_gp_points_in_bbox(const point3f &lim_min, const point3f &lim_max,
                                         vector<int> &out) {
        float a_min[] = {lim_min.x(), lim_min.y(), lim_min.z()};
        float a_max[] = {lim_max.x(), lim_max.y(), lim_max.z()};
        return rtree.Search(a_min, a_max, BGKLOctoMap::search_callback, static_cast<void *>(&out));
    }

    int BGKLOctoMap::has_gp_points_in_bbox(const point3f &lim_min,
                                         const point3f &lim_max) {
        float a_min[] = {lim_min.x(), lim_min.y(), lim_min.z()};
        float a_max[] = {lim_max.x(), lim_max.y(), lim_max.z()};
        return rtree.Search(a_min, a_max, BGKLOctoMap::count_callback, NULL);
    }

    bool BGKLOctoMap::count_callback(int k, void *arg) {
        return false;
    }

    bool BGKLOctoMap::search_callback(int k, void *arg) {
        // GPLineCloud *out = static_cast<GPLineCloud *>(arg);
        vector<int> *out = static_cast<vector<int> *>(arg);
        out->push_back(k);
        return true;
    }


    int BGKLOctoMap::has_gp_points_in_bbox(const ExtendedBlock &block) {
        for (auto it = block.cbegin(); it != block.cend(); ++it) {
            if (has_gp_points_in_bbox(*it) > 0)
                return 1;
        }
        return 0;
    }

    int BGKLOctoMap::get_gp_points_in_bbox(const ExtendedBlock &block,
                                         vector<int> &out) {
        int n = 0;
        for (auto it = block.cbegin(); it != block.cend(); ++it) {
            n += get_gp_points_in_bbox(*it, out);
        }
        return n;
    }

    Block *BGKLOctoMap::search(BlockHashKey key) const {
        auto block = block_arr.find(key);
        if (block == block_arr.end()) {
            return nullptr;
        } else {
            return block->second;
        }
    }

    OcTreeNode BGKLOctoMap::search(point3f p) const {
        Block *block = search(block_to_hash_key(p));
        if (block == nullptr) {
            return OcTreeNode();
        } else {
            return OcTreeNode(block->search(p));
        }
    }

    OcTreeNode BGKLOctoMap::search(float x, float y, float z) const {
        return search(point3f(x, y, z));
    }
}
