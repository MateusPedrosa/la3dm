#ifndef LA3DM_BGKL_OCCUPANCY_H
#define LA3DM_BGKL_OCCUPANCY_H

#include <iostream>
#include <fstream>
#include <cmath>
#include "point3f.h"

namespace la3dm {

    /// Occupancy state: before pruning: FREE, OCCUPIED, UNKNOWN; after pruning: PRUNED
    enum class State : char {
        FREE, OCCUPIED, UNCERTAIN, UNKNOWN, PRUNED
    };

    /*
     * @brief Inference outputs and occupancy state.
     *
     * Occupancy has member variables: m_A and m_B (kernel densities of positive
     * and negative class, respectively) and State.
     * Before using this class, set the static member variables first.
     *
     * Each voxel also maintains a per-voxel 3x3 information matrix (stored as
     * upper triangle) that tracks the directional constraint diversity of past
     * observations. This is used to compute per-voxel evidence weights that
     * prevent false confidence accumulation from redundant viewpoints.
     */
    class Occupancy {
        friend std::ostream &operator<<(std::ostream &os, const Occupancy &oc);

        friend std::ofstream &operator<<(std::ofstream &os, const Occupancy &oc);

        friend std::ifstream &operator>>(std::ifstream &is, Occupancy &oc);

        friend class BGKLOctoMap;

    public:
        /*
         * @brief Constructors and destructor.
         */
        Occupancy() : m_A(Occupancy::prior_A), m_B(Occupancy::prior_B),
                      state(State::UNKNOWN), classified(false), is_trusted(false),
                      lambda_max_cache(0.0f), info_constrained(false) {
            info[0] = info[1] = info[2] = info[3] = info[4] = info[5] = 0.0f;
        }

        Occupancy(float A, float B);

        Occupancy(const Occupancy &other)
            : m_A(other.m_A), m_B(other.m_B), state(other.state),
              classified(other.classified), is_trusted(other.is_trusted),
              lambda_max_cache(other.lambda_max_cache),
              info_constrained(other.info_constrained) {
            for (int i = 0; i < 6; ++i) info[i] = other.info[i];
        }

        Occupancy &operator=(const Occupancy &other) {
            m_A = other.m_A;
            m_B = other.m_B;
            state = other.state;
            classified = other.classified;
            is_trusted = other.is_trusted;
            lambda_max_cache = other.lambda_max_cache;
            info_constrained = other.info_constrained;
            for (int i = 0; i < 6; ++i) info[i] = other.info[i];
            return *this;
        }

        ~Occupancy() { }

        /*
         * @brief Exact updates for nonparametric Bayesian kernel inference
         * @param ybar kernel density estimate of positive class (occupied)
         * @param kbar kernel density of negative class (unoccupied)
         * @param obs_range observation range (distance from sensor to node at the time of observation)
         */
        void update(float ybar, float kbar, float obs_range);

        /// Get probability of occupancy.
        float get_prob() const;

        inline float get_A() const { return m_A; }
        inline float get_B() const { return m_B; }

        /// Get variance of occupancy (uncertainty)
        inline float get_var() const { return (m_A * m_B) / ( (m_A + m_B) * (m_A + m_B) * (m_A + m_B + 1.0f)); }

        /*
         * @brief Get occupancy state of the node.
         * @return occupancy state (see State).
         */
        inline State get_state() const { return state; }

        /// Prune current node; set state to PRUNED.
        inline void prune() { state = State::PRUNED; }

        /// Only FREE and OCCUPIED nodes can be equal.
        inline bool operator==(const Occupancy &rhs) const {
            return this->state != State::UNKNOWN && this->state == rhs.state;
        }

        /*
         * @brief Compute per-voxel novelty weight for a new observation.
         *
         * w_voxel = max(0, 1 - (nᵀ I(p) n) / W_sat)   where W_sat = tau_info
         *
         * Returns 1.0 for new voxels (zero info matrix → full evidence weight).
         * Returns 0.0 if voxel is marked as well-constrained (info_constrained).
         *
         * @param n  Unit constraint direction vector in world frame.
         */
        inline float compute_w_novelty(const point3f &n) const {
            if (info_constrained) return 0.0f;
            float nIn = info[0]*n.x()*n.x() + info[3]*n.y()*n.y() + info[5]*n.z()*n.z()
                      + 2.0f*info[1]*n.x()*n.y() + 2.0f*info[2]*n.x()*n.z()
                      + 2.0f*info[4]*n.y()*n.z();
            float w = 1.0f - nIn / (tau_info + 1e-6f);
            return (w < 0.0f) ? 0.0f : ((w > 1.0f) ? 1.0f : w);
        }

        /*
         * @brief Rank-1 update of the information matrix.
         *
         * I(p) ← I(p) + w · n nᵀ
         * lambda_max_cache ← trace(I)  (upper bound on λ_max)
         *
         * No-op if voxel is marked as well-constrained.
         *
         * @param n  Unit constraint direction vector in world frame.
         * @param w  Gated evidence weight w_total = w_r * w_voxel for this observation.
         */
        inline void update_info_matrix(const point3f &n, float w) {
            if (info_constrained) return;
            info[0] += w * n.x() * n.x();
            info[1] += w * n.x() * n.y();
            info[2] += w * n.x() * n.z();
            info[3] += w * n.y() * n.y();
            info[4] += w * n.y() * n.z();
            info[5] += w * n.z() * n.z();
            lambda_max_cache = info[0] + info[3] + info[5];  // trace = upper bound on λ_max
        }

        /*
         * @brief Lifecycle management: mark voxel as well-constrained when
         *        Beta variance < tau_var AND lam2 of the projected 2×2 I > tau_info.
         *
         * Uses the most-recent observation's los direction to project I onto the
         * 2D plane ⊥ los, then checks the minimum (weak) eigenvalue of that 2×2
         * projection. This correctly handles the los-direction nullspace that makes
         * the raw 3×3 minimum eigenvalue always near zero.
         *
         * @param los_hat  Unit line-of-sight vector from sensor to voxel (world frame).
         */
        void check_deallocation(const point3f &los_hat);

        /*
         * @brief Compute the 2D eigenstructure of I projected onto the plane ⊥ los.
         *
         * Projects the 3×3 information matrix onto an orthonormal basis of the plane
         * perpendicular to los, then analytically decomposes the resulting 2×2 matrix.
         * Returns lam1 (dominant), lam2 (weak), and v_weak (world-frame unit vector
         * pointing along the least-observed arc direction).
         *
         * @param los    Unit line-of-sight direction (world frame).
         * @param lam1   Output: dominant eigenvalue of projected 2×2 matrix.
         * @param lam2   Output: weak eigenvalue of projected 2×2 matrix.
         * @param v_weak Output: world-frame unit weak eigenvector (⊥ los).
         */
        void get_2d_eigenstruct(const point3f &los,
                                float &lam1, float &lam2,
                                point3f &v_weak) const;

        /// @return true if info matrix is still active (voxel not yet well-constrained).
        inline bool has_active_info_matrix() const { return !info_constrained; }

        /// @return the cached lambda_max (trace of I) for external queries.
        inline float get_lambda_max_cache() const { return lambda_max_cache; }

        /// Copy the raw information matrix (upper-triangle [Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
        /// out for deferred eigenstruct computation after releasing the map lock.
        inline void get_info(float out[6]) const {
            for (int i = 0; i < 6; ++i) out[i] = info[i];
        }

        /// Compute 2D eigenstructure from a raw info array — no node instance required.
        /// Identical to get_2d_eigenstruct() but callable without holding a node reference.
        static void compute_2d_eigenstruct_raw(const float info[6], const point3f &los,
                                               float &lam1, float &lam2, point3f &v_weak);

        bool classified;

        // Public so they can be set from external code (e.g. the ROS server node)
        // before any voxels are updated.
        static float tau_var;   // Beta variance threshold for lifecycle deallocation (default 0.01)
        static float tau_info;  // W_sat: saturation constant for novelty weight and deallocation (default 0.5)
        static float delta;     // deallocation tolerance: constrained when lambda2 >= (1-delta)*tau_info (default 0.05)

        inline bool get_is_trusted() const { return is_trusted; }

    private:
        float m_A;
        float m_B;
        bool is_trusted{false};
        State state;

        // Per-voxel information matrix (upper triangle of 3×3 symmetric matrix, world frame)
        // Layout: [Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
        float info[6];
        float lambda_max_cache;  // = trace(I); upper bound on λ_max; updated on every rank-1 update
        bool info_constrained;   // true when voxel is well-constrained (info matrix logically deallocated)

        static float sf2;
        static float ell;   // length-scale

        static float prior_A; // prior on alpha
        static float prior_B; // prior on beta

        static float free_thresh;     // FREE occupancy threshold
        static float occupied_thresh; // OCCUPIED occupancy threshold
        static float var_thresh;
    };

    typedef Occupancy OcTreeNode;
}

#endif // LA3DM_BGKL_OCCUPANCY_H
