#include <algorithm>
#include <cmath>
#include <limits>
#include <ros/ros.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Dense>
#include "bgksoctomap.h"
#include "bgksinference.h"

using std::vector;

// #define DEBUG true;

#ifdef DEBUG

#include <iostream>

#define Debug_Msg(msg) {\
std::cout << "Debug: " << msg << std::endl; }
#endif

namespace la3dm {

    BGKSOctoMap::BGKSOctoMap() : BGKSOctoMap(0.1f, // resolution
                                        4, // block_depth
                                        1.0, // sf2
                                        1.0, // ell
                                        0.3f, // free_thresh
                                        0.7f, // occupied_thresh
                                        1.0f, // var_thresh
                                        1.0f, // prior_A
                                        1.0f // prior_B
                                    ) { }

    BGKSOctoMap::BGKSOctoMap(float resolution,
                        unsigned short block_depth,
                        float sf2,
                        float ell,
                        float free_thresh,
                        float occupied_thresh,
                        float var_thresh,
                        float prior_A,
                        float prior_B,
                        float theta_bw,
                        float phi_bw,
                        bool free_ray_range_weight)
            : resolution(resolution), block_depth(block_depth),
              block_size((float) pow(2, block_depth - 1) * resolution),
              theta_bw(theta_bw), phi_bw(phi_bw),
              free_ray_range_weight(free_ray_range_weight),
              pose_level_weighting_(false),
              pose_history_K_(20),
              pose_novelty_sigma_(0.3f) {
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

    BGKSOctoMap::~BGKSOctoMap() {
        for (auto it = block_arr.begin(); it != block_arr.end(); ++it) {
            if (it->second != nullptr) {
                delete it->second;
            }
        }
    }

    void BGKSOctoMap::configure_pose_level_weighting(
            bool enabled, int K, float sigma,
            float w_roll, float w_pitch, float w_yaw,
            float w_vx_l2, float w_vy_l2, float w_vz_l2) {
        pose_level_weighting_ = enabled;
        pose_history_K_       = K;
        pose_novelty_sigma_   = sigma;
        pose_w_[0] = w_roll;
        pose_w_[1] = w_pitch;
        pose_w_[2] = w_yaw;
        pose_w_[3] = w_vx_l2;
        pose_w_[4] = w_vy_l2;
        pose_w_[5] = w_vz_l2;
    }

    void BGKSOctoMap::configure_frustum(float swath_angle_deg) {
        if (swath_angle_deg <= 0.0f)
            swath_half_angle_ = (float)M_PI;  // disabled — no azimuth culling
        else
            swath_half_angle_ = (swath_angle_deg * 0.5f) * (float)M_PI / 180.0f;
    }

    void BGKSOctoMap::update_pose_history_(float px, float py, float pz,
                                           float qx, float qy, float qz, float qw) {
        PoseEntry e;
        e.px = px; e.py = py; e.pz = pz;
        e.qx = qx; e.qy = qy; e.qz = qz; e.qw = qw;
        pose_history_.push_back(e);
        while ((int)pose_history_.size() > pose_history_K_)
            pose_history_.pop_front();
    }

    float BGKSOctoMap::compute_w_pose_(float px, float py, float pz,
                                        float qx, float qy, float qz, float qw) const {
        if (pose_history_.empty()) return 1.0f;

        Eigen::Quaternionf q_cur(qw, qx, qy, qz);
        Eigen::Vector3f    p_cur(px, py, pz);

        float d_min_sq = std::numeric_limits<float>::max();

        for (const auto& entry : pose_history_) {
            Eigen::Quaternionf q_i(entry.qw, entry.qx, entry.qy, entry.qz);
            Eigen::Vector3f    p_i(entry.px, entry.py, entry.pz);

            // Relative rotation: q_cur expressed in frame of q_i
            Eigen::Quaternionf q_rel = q_cur * q_i.conjugate();
            q_rel.normalize();

            // SO(3) log map via AngleAxisf (handles near-identity correctly: 0*arbitrary_axis = 0)
            Eigen::AngleAxisf aa(q_rel);
            Eigen::Vector3f omega = aa.axis() * aa.angle();

            // Relative translation in body frame of pose i
            Eigen::Matrix3f R_i = q_i.toRotationMatrix();
            Eigen::Vector3f t_body = R_i.transpose() * (p_cur - p_i);

            float xi[6] = { omega[0], omega[1], omega[2],
                            t_body[0], t_body[1], t_body[2] };

            float d_sq = 0.0f;
            for (int k = 0; k < 6; ++k)
                d_sq += pose_w_[k] * xi[k] * xi[k];

            if (d_sq < d_min_sq)
                d_min_sq = d_sq;
        }

        float sigma2 = pose_novelty_sigma_ * pose_novelty_sigma_;
        float w = 1.0f - std::exp(-d_min_sq / sigma2);
        if (w < 0.0f) w = 0.0f;
        if (w > 1.0f) w = 1.0f;
        return w;
    }

    void BGKSOctoMap::set_resolution(float resolution) {
        this->resolution = resolution;
        Block::resolution = resolution;
        this->block_size = (float) pow(2, block_depth - 1) * resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
    }

    void BGKSOctoMap::set_block_depth(unsigned short max_depth) {
        this->block_depth = max_depth;
        OcTree::max_depth = max_depth;
        this->block_size = (float) pow(2, block_depth - 1) * resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
    }

    void BGKSOctoMap::insert_pointcloud(const PCLPointCloud &cloud, const point3f &origin,
                                      const point3f &sensor_up,
                                      float ds_resolution,
                                      float free_res, float max_range,
                                      float qx, float qy, float qz, float qw) {

#ifdef DEBUG
        Debug_Msg("Insert pointcloud: " << "cloud size: " << cloud.size() << " origin: " << origin);
#endif

        // ---- Per-frame pose novelty weight ----
        float w_pose_frame = 1.0f;
        if (pose_level_weighting_) {
            w_pose_frame = compute_w_pose_(origin.x(), origin.y(), origin.z(), qx, qy, qz, qw);
            update_pose_history_(origin.x(), origin.y(), origin.z(), qx, qy, qz, qw);
            ROS_INFO_THROTTLE(1.0, "[pose_novelty] w_pose=%.4f  history=%zu",
                              (double)w_pose_frame, pose_history_.size());
        }

        // Derive sensor forward direction (X-axis) from quaternion
        point3f forward_hat(
            1.0f - 2.0f*(qy*qy + qz*qz),
            2.0f*(qx*qy + qz*qw),
            2.0f*(qx*qz - qy*qw)
        );

        ////////// Preparation //////////////////////////
        /////////////////////////////////////////////////
        GPLineCloud xy;
        GPLineCloud rays;
        vector<int> ray_idx;
        get_training_data(cloud, origin, ds_resolution, free_res, max_range, xy, rays, ray_idx);
        assert (ray_idx.size() == xy.size());

        if (xy.empty()) {
            return; 
        }

        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);

        vector<BlockHashKey> blocks;
        get_blocks_in_bbox(lim_min, lim_max, blocks, origin, sensor_up, forward_hat);

        // std::unordered_map<BlockHashKey, GPLineCloud> key_train_data_map;
        for (int k = 0; k < xy.size(); ++k) {
            float p[] = {xy[k].first.x0(), xy[k].first.y0(), xy[k].first.z0()};
            rtree.Insert(p, p, k);

        }
        /////////////////////////////////////////////////

        ////////// Training /////////////////////////////
        /////////////////////////////////////////////////
        vector<BlockHashKey> test_blocks;
        std::unordered_map<BlockHashKey, BGKS3f *> bgks_arr;

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
                    block_w.push_back(rays[ray_idx[xy_idx[j]]].second); // range-based weight, same as occupied
                }
            }
            };
            // std::cout << "number of training blocks" << block_y.size() << std::endl;
            BGKS3f *bgks = new BGKS3f(OcTreeNode::sf2, OcTreeNode::ell, theta_bw, phi_bw);
            bgks->train(block_x, block_y, block_w);
#ifdef OPENMP
#pragma omp critical
#endif
            {
                bgks_arr.emplace(key, bgks);
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
                auto bgks = bgks_arr.find(*block_it);
                if (bgks == bgks_arr.end())
                    continue;

                vector<float> ybar, kbar;
                bgks->second->predict(origin, sensor_up, xs, ybar, kbar);

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
                        point3f los_hat;
                        point3f n_vec;
                        bool valid_n = false;

                        if (obs_range > 1e-6f) {
                            los_hat = los_vec * (1.0f / obs_range);
                            float dot_su = (float)sensor_up.dot(los_hat);
                            n_vec = sensor_up - los_hat * dot_su;
                            float n_norm = (float)n_vec.norm();
                            if (n_norm > 1e-8f) {
                                n_vec *= (1.0f / n_norm);
                                valid_n = true;
                            }
                        }

                        if (debug_voxel_enabled_ &&
                            std::fabs(node_loc.x() - debug_voxel_x_) < 0.06f &&
                            std::fabs(node_loc.y() - debug_voxel_y_) < 0.06f &&
                            std::fabs(node_loc.z() - debug_voxel_z_) < 0.06f) {
                            float inf[6]; node.get_info(inf);
                            float nIn = 0.0f;
                            if (valid_n) {
                                const point3f &n = n_vec;
                                nIn = inf[0]*n.x()*n.x() + inf[3]*n.y()*n.y() + inf[5]*n.z()*n.z()
                                    + 2.0f*inf[1]*n.x()*n.y() + 2.0f*inf[2]*n.x()*n.z()
                                    + 2.0f*inf[4]*n.y()*n.z();
                            }
                            float wv = valid_n ? node.compute_w_novelty(n_vec) : 1.0f;
                            float wr = std::max(0.05f, 1.0f / (obs_range + 1.0f));
                            ROS_INFO_STREAM("[dbg_voxel] "
                                << "kbar=" << kbar[j] << " ybar=" << ybar[j]
                                << " range=" << obs_range
                                << " nIn=" << nIn << " tau_info=" << OcTreeNode::tau_info
                                << " w_voxel=" << wv << " w_range=" << wr
                                << " mA=" << node.get_A() << " mB=" << node.get_B()
                                << " trusted=" << node.get_is_trusted()
                                << " var=" << node.get_var());
                        }

                        if (ablate_directional_weights_) {
                            // Ablation: raw unweighted Beta accumulation, no info matrix.
                            node.update(ybar[j], kbar[j], obs_range);
                        } else if (valid_n) {
                            // Capture new-voxel flag BEFORE info matrix update
                            bool is_new_voxel = (node.get_lambda_max_cache() < 1e-9f);

                            float w_voxel = node.compute_w_novelty(n_vec);  // pre-update

                            float w_range = 1.0f / (obs_range + 1.0f);
                            if (w_range < 0.05f) w_range = 0.05f;
                            float w_total = w_range * w_voxel;
                            node.update_info_matrix(n_vec, w_total);
                            node.check_deallocation(los_hat);

                            float w_beta = pose_level_weighting_
                                           ? (is_new_voxel ? 1.0f : w_pose_frame)
                                           : w_total;
                            if (w_beta > 1e-8f)
                                node.update(ybar[j] * w_beta, kbar[j] * w_beta, obs_range);
                        } else {
                            // Degenerate geometry: LOS parallel to sensor_up (rare).
                            float w_beta = pose_level_weighting_ ? w_pose_frame : 1.0f;
                            node.update(ybar[j] * w_beta, kbar[j] * w_beta, obs_range);
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
        for (auto it = bgks_arr.begin(); it != bgks_arr.end(); ++it) {
            delete it->second;
        }
        // ray_keys.clear();


        rtree.RemoveAll();
    }

    BGKSPreparedUpdate::~BGKSPreparedUpdate() {
        for (auto it = bgks_arr.begin(); it != bgks_arr.end(); ++it)
            delete it->second;
    }

    BGKSPreparedUpdate BGKSOctoMap::prepare_pointcloud_update(const PCLPointCloud &cloud,
                                                               const point3f &origin,
                                                               const point3f &sensor_up,
                                                               float ds_resolution,
                                                               float free_res,
                                                               float max_range,
                                                               float qx, float qy,
                                                               float qz, float qw) {
        BGKSPreparedUpdate result;
        result.origin     = origin;
        result.sensor_up  = sensor_up;

        // ---- Per-frame pose novelty weight ----
        float w_pose_frame = 1.0f;
        if (pose_level_weighting_) {
            w_pose_frame = compute_w_pose_(origin.x(), origin.y(), origin.z(), qx, qy, qz, qw);
            update_pose_history_(origin.x(), origin.y(), origin.z(), qx, qy, qz, qw);
            ROS_INFO_THROTTLE(1.0, "[pose_novelty] w_pose=%.4f  history=%zu",
                              (double)w_pose_frame, pose_history_.size());
        }
        result.w_pose = w_pose_frame;

        // Derive sensor forward direction (X-axis) from quaternion
        point3f forward_hat(
            1.0f - 2.0f*(qy*qy + qz*qz),
            2.0f*(qx*qy + qz*qw),
            2.0f*(qx*qz - qy*qw)
        );

        ////////// Preparation //////////////////////////
        GPLineCloud xy;
        GPLineCloud rays;
        vector<int> ray_idx;
        get_training_data(cloud, origin, ds_resolution, free_res, max_range, xy, rays, ray_idx);
        assert(ray_idx.size() == xy.size());

        if (xy.empty())
            return result;  // result.empty == true

        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);

        vector<BlockHashKey> blocks;
        get_blocks_in_bbox(lim_min, lim_max, blocks, origin, sensor_up, forward_hat);

        for (int k = 0; k < (int)xy.size(); ++k) {
            float p[] = {xy[k].first.x0(), xy[k].first.y0(), xy[k].first.z0()};
            rtree.Insert(p, p, k);
        }
        /////////////////////////////////////////////////

        ////////// Training /////////////////////////////
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < (int)blocks.size(); ++i) {
            BlockHashKey key = blocks[i];
            ExtendedBlock eblock = get_extended_block(key);
            if (has_gp_points_in_bbox(eblock))
#ifdef OPENMP
#pragma omp critical
#endif
            {
                result.test_blocks.push_back(key);
            };

            vector<int> xy_idx;
            get_gp_points_in_bbox(key, xy_idx);
            if (xy_idx.size() < 1)
                continue;

            vector<int> ray_keys(rays.size(), 0);
            vector<float> block_x, block_y, block_w;
            for (int j = 0; j < (int)xy_idx.size(); ++j) {
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
                    block_w.push_back(xy[xy_idx[j]].second);
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
                    block_w.push_back(rays[ray_idx[xy_idx[j]]].second);
                }
            }
            };
            BGKS3f *bgks = new BGKS3f(OcTreeNode::sf2, OcTreeNode::ell, theta_bw, phi_bw);
            bgks->train(block_x, block_y, block_w);
#ifdef OPENMP
#pragma omp critical
#endif
            {
                result.bgks_arr.emplace(key, bgks);
            };
        }
        /////////////////////////////////////////////////

        rtree.RemoveAll();
        result.empty = false;
        return result;
    }

    void BGKSOctoMap::commit_pointcloud_update(const BGKSPreparedUpdate &upd,
                                               std::vector<DirtyEntry>* dirty_out) {
        if (upd.empty) return;

        ////////// Prediction ///////////////////////////
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < (int)upd.test_blocks.size(); ++i) {
            BlockHashKey key = upd.test_blocks[i];
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
                auto bgks = upd.bgks_arr.find(*block_it);
                if (bgks == upd.bgks_arr.end())
                    continue;

                vector<float> ybar, kbar;
                bgks->second->predict(upd.origin, upd.sensor_up, xs, ybar, kbar);

                int j = 0;
                for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it, ++j) {
                    OcTreeNode &node = leaf_it.get_node();
                    auto node_loc = block->get_loc(leaf_it);

                    float obs_range = (float)(node_loc - upd.origin).norm();

                    if (kbar[j] > 0.000001f) {
                        point3f los_vec = node_loc - upd.origin;
                        point3f los_hat;
                        point3f n_vec;
                        bool valid_n = false;

                        if (obs_range > 1e-6f) {
                            los_hat = los_vec * (1.0f / obs_range);
                            float dot_su = (float)upd.sensor_up.dot(los_hat);
                            n_vec = upd.sensor_up - los_hat * dot_su;
                            float n_norm = (float)n_vec.norm();
                            if (n_norm > 1e-8f) {
                                n_vec *= (1.0f / n_norm);
                                valid_n = true;
                            }
                        }

                        if (ablate_directional_weights_) {
                            // Ablation: raw unweighted Beta accumulation, no info matrix.
                            node.update(ybar[j], kbar[j], obs_range);

                            if (dirty_out) {
                                DirtyEntry e;
                                e.pos      = node_loc;
                                e.priority = node.get_var();
                                e.active   = e.priority > 1e-8f;
                                e.state    = node.get_state();
                                node.get_info(e.info);
#ifdef OPENMP
                                #pragma omp critical
#endif
                                dirty_out->push_back(e);
                            }
                        } else if (valid_n) {
                            bool is_new_voxel = (node.get_lambda_max_cache() < 1e-9f);

                            float w_voxel = node.compute_w_novelty(n_vec);  // pre-update

                            float w_range = 1.0f / (obs_range + 1.0f);
                            if (w_range < 0.05f) w_range = 0.05f;
                            float w_total = w_range * w_voxel;
                            node.update_info_matrix(n_vec, w_total);
                            node.check_deallocation(los_hat);

                            float w_beta = pose_level_weighting_
                                           ? (is_new_voxel ? 1.0f : upd.w_pose)
                                           : w_total;
                            if (w_beta > 1e-8f)
                                node.update(ybar[j] * w_beta, kbar[j] * w_beta, obs_range);

                            if (dirty_out) {
                                DirtyEntry e;
                                e.pos      = node_loc;
                                e.priority = node.get_var();
                                e.active   = e.priority > 1e-8f;
                                e.state    = node.get_state();
                                node.get_info(e.info);
#ifdef OPENMP
                                #pragma omp critical
#endif
                                dirty_out->push_back(e);
                            }
                        } else {
                            float w_beta = pose_level_weighting_ ? upd.w_pose : 1.0f;
                            node.update(ybar[j] * w_beta, kbar[j] * w_beta, obs_range);
                        }
                    }
                }
            }
        }
        /////////////////////////////////////////////////
    }

    void BGKSOctoMap::get_bbox(point3f &lim_min, point3f &lim_max) const {
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

    void BGKSOctoMap::get_training_data(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
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

                // if (peak_weight < 0.05f) peak_weight = 0.05f; // minimum floor
                
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
                    float ray_weight = free_ray_range_weight ? 1.0f / (true_dist * true_dist) : 1.0f;
                    rays.emplace_back(line6f, ray_weight);
                }
            }
        }
    }

    void BGKSOctoMap::downsample(const PCLPointCloud &in, PCLPointCloud &out, float ds_resolution) const {
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

    void BGKSOctoMap::beam_sample(const point3f &hit, const point3f &origin, PointCloud &frees,
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

    void BGKSOctoMap::bbox(const GPLineCloud &cloud, point3f &lim_min, point3f &lim_max) const {
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

    void BGKSOctoMap::get_blocks_in_bbox(const point3f &lim_min, const point3f &lim_max,
                                   vector<BlockHashKey> &blocks,
                                   const point3f &origin,
                                   const point3f &sensor_up,
                                   const point3f &forward_hat) const {
        // Symmetrical 2-block padding to support wide vertical apertures
        float pad = 2.0f * block_size;

        // Precompute frustum limits.
        // Elevation: always on (phi_bw / 2).  Azimuth: on when swath_half_angle_ < π.
        const bool do_azimuth = (swath_half_angle_ < (float)M_PI - 1e-4f);

        // Project forward_hat onto the swath plane (perpendicular to sensor_up).
        // Done once per scan, not per block.
        point3f fwd_horiz;
        float fwd_horiz_norm = 0.0f;
        if (do_azimuth) {
            fwd_horiz = forward_hat - sensor_up * forward_hat.dot(sensor_up);
            fwd_horiz_norm = fwd_horiz.norm();
        }

        for (float x = lim_min.x() - pad; x <= lim_max.x() + pad; x += block_size) {
            for (float y = lim_min.y() - pad; y <= lim_max.y() + pad; y += block_size) {
                for (float z = lim_min.z() - pad; z <= lim_max.z() + pad; z += block_size) {

                    // ---- Frustum culling ----
                    point3f bc(x, y, z);
                    point3f v = bc - origin;
                    float dist = v.norm();

                    if (dist > 1e-3f) {
                        // Angular slack = solid angle subtended by half-diagonal of block
                        float slack = std::atan2(block_size * 0.866f, dist);

                        // Elevation test (always active)
                        float sin_elev = v.dot(sensor_up) / dist;
                        if (std::abs(sin_elev) > std::sin(phi_bw * 0.5f + slack))
                            continue;

                        // Azimuth test (active when swath_angle configured)
                        if (do_azimuth && fwd_horiz_norm > 1e-6f) {
                            point3f v_horiz = v - sensor_up * v.dot(sensor_up);
                            float v_horiz_norm = v_horiz.norm();
                            if (v_horiz_norm > 1e-6f) {
                                float cos_az = v_horiz.dot(fwd_horiz) /
                                               (v_horiz_norm * fwd_horiz_norm);
                                if (cos_az < std::cos(swath_half_angle_ + slack))
                                    continue;
                            }
                        }
                    }

                    blocks.push_back(block_to_hash_key(x, y, z));
                }
            }
        }
    }

    int BGKSOctoMap::get_gp_points_in_bbox(const BlockHashKey &key,
                                         vector<int> &out) {
        point3f half_size(block_size / 2.0f, block_size / 2.0f, block_size / 2.0);
        point3f lim_min = hash_key_to_block(key) - half_size;
        point3f lim_max = hash_key_to_block(key) + half_size;
        return get_gp_points_in_bbox(lim_min, lim_max, out);
    }

    int BGKSOctoMap::has_gp_points_in_bbox(const BlockHashKey &key) {
        point3f half_size(block_size / 2.0f, block_size / 2.0f, block_size / 2.0);
        point3f lim_min = hash_key_to_block(key) - half_size;
        point3f lim_max = hash_key_to_block(key) + half_size;
        return has_gp_points_in_bbox(lim_min, lim_max);
    }

    int BGKSOctoMap::get_gp_points_in_bbox(const point3f &lim_min, const point3f &lim_max,
                                         vector<int> &out) {
        float a_min[] = {lim_min.x(), lim_min.y(), lim_min.z()};
        float a_max[] = {lim_max.x(), lim_max.y(), lim_max.z()};
        return rtree.Search(a_min, a_max, BGKSOctoMap::search_callback, static_cast<void *>(&out));
    }

    int BGKSOctoMap::has_gp_points_in_bbox(const point3f &lim_min,
                                         const point3f &lim_max) {
        float a_min[] = {lim_min.x(), lim_min.y(), lim_min.z()};
        float a_max[] = {lim_max.x(), lim_max.y(), lim_max.z()};
        return rtree.Search(a_min, a_max, BGKSOctoMap::count_callback, NULL);
    }

    bool BGKSOctoMap::count_callback(int k, void *arg) {
        return false;
    }

    bool BGKSOctoMap::search_callback(int k, void *arg) {
        // GPLineCloud *out = static_cast<GPLineCloud *>(arg);
        vector<int> *out = static_cast<vector<int> *>(arg);
        out->push_back(k);
        return true;
    }


    int BGKSOctoMap::has_gp_points_in_bbox(const ExtendedBlock &block) {
        for (auto it = block.cbegin(); it != block.cend(); ++it) {
            if (has_gp_points_in_bbox(*it) > 0)
                return 1;
        }
        return 0;
    }

    int BGKSOctoMap::get_gp_points_in_bbox(const ExtendedBlock &block,
                                         vector<int> &out) {
        int n = 0;
        for (auto it = block.cbegin(); it != block.cend(); ++it) {
            n += get_gp_points_in_bbox(*it, out);
        }
        return n;
    }

    Block *BGKSOctoMap::search(BlockHashKey key) const {
        auto block = block_arr.find(key);
        if (block == block_arr.end()) {
            return nullptr;
        } else {
            return block->second;
        }
    }

    OcTreeNode BGKSOctoMap::search(point3f p) const {
        Block *block = search(block_to_hash_key(p));
        if (block == nullptr) {
            return OcTreeNode();
        } else {
            return OcTreeNode(block->search(p));
        }
    }

    OcTreeNode BGKSOctoMap::search(float x, float y, float z) const {
        return search(point3f(x, y, z));
    }
}
