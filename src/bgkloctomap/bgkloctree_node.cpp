#include "bgkloctree_node.h"
#include <cmath>
#include <algorithm>

namespace la3dm {

    /// Default static values
    float Occupancy::sf2 = 1.0f;
    float Occupancy::ell = 1.0f;
    float Occupancy::free_thresh = 0.3f;
    float Occupancy::occupied_thresh = 0.7f;
    float Occupancy::var_thresh = 1000.0f;
    float Occupancy::prior_A = 0.5f;
    float Occupancy::prior_B = 0.5f;
    Occupancy::Occupancy(float A, float B)
        : m_A(Occupancy::prior_A + A), m_B(Occupancy::prior_B + B),
          classified(false), is_trusted(false),
          lambda_max_cache(0.0f) {
        info[0] = info[1] = info[2] = info[3] = info[4] = info[5] = 0.0f;
        float var = get_var();
        if (var > Occupancy::var_thresh)
            state = State::UNKNOWN;
        else {
            float p = get_prob();
            state = p > Occupancy::occupied_thresh ? State::OCCUPIED : (p < Occupancy::free_thresh ? State::FREE
                                                                                                   : State::UNKNOWN);
        }
    }

    float Occupancy::get_prob() const {
        return m_A / (m_A + m_B);
    }

    void Occupancy::update(float ybar, float kbar, float obs_range) {
        classified = true;

        const float R = 40.0f;

        float occ_w = ybar;
        float free_w = kbar - ybar;

        if (obs_range < R) {
            if (!is_trusted) {
                m_A = 0.0f;         // Reset the accumulated uncertain mass
                // m_B = 0.0f;         // Reset the accumulated free mass
                is_trusted = true;  // Promote voxel to trusted
            }
            m_A += occ_w;           // Accumulate trusted evidence
            m_B += free_w;          // Accumulate free evidence
        }
        else {
            // We are far away, this is uncertain evidence
            if (!is_trusted) {
                m_A += occ_w;       // Accumulate uncertain mass
                m_B += free_w;      // Accumulate free evidence
            } else {
                // Pass. Voxel is already trusted, protect the estimate
                // from noisy long-range observations.
            }
        }

        float var = get_var();
        if (var > Occupancy::var_thresh)
            state = State::UNKNOWN;
        else {
            float p = get_prob();
            state = p > Occupancy::occupied_thresh ? (is_trusted ? State::OCCUPIED : State::UNCERTAIN) :
                                                     (p < Occupancy::free_thresh ? State::FREE : State::UNKNOWN);
        }
    }

    // ---------------------------------------------------------------------------
    // Per-voxel information matrix lifecycle management
    // ---------------------------------------------------------------------------

    // Project the 3×3 information matrix I (upper-triangle [Ixx,Ixy,Ixz,Iyy,Iyz,Izz])
    // onto the 2D plane ⊥ los, then analytically decompose the resulting 2×2 symmetric
    // matrix to obtain eigenvalues and the weak eigenvector.
    //
    // The los-direction nullspace of I (always near-zero eigenvalue) is thus discarded,
    // and only the 2D subspace that carries planning-relevant information is examined.
    // This matches §4–§5 of the NBV Planner Implementation Specification.
    static void compute_2d_eigenstruct_impl(
            const float I[6], const point3f &los,
            float &lam1, float &lam2, point3f &v_weak)
    {
        // --- Build orthonormal basis (e1, e2) for the plane ⊥ los ---
        // Numerically stable: fall back to world_x when los is nearly vertical.
        point3f world_z(0.0f, 0.0f, 1.0f);
        point3f world_x(1.0f, 0.0f, 0.0f);
        point3f e1;
        if (std::fabs(los.dot(world_z)) < 0.9f)
            e1 = los.cross(world_z);
        else
            e1 = los.cross(world_x);
        float e1_norm = (float)e1.norm();
        if (e1_norm < 1e-8f) { e1 = world_x; e1_norm = 1.0f; }
        e1 *= (1.0f / e1_norm);
        point3f e2 = los.cross(e1);  // already unit: los⊥e1, both unit

        // --- Project 3×3 onto {e1, e2} to get 2×2 matrix [a, b; b, c] ---
        // I_2d[i,j] = ei^T * I * ej  (I symmetric, stored as upper triangle)
        auto quadform = [&](const point3f &u, const point3f &v) -> float {
            // u^T * I * v  for symmetric I stored as [Ixx,Ixy,Ixz,Iyy,Iyz,Izz]
            float Iv_x = I[0]*v.x() + I[1]*v.y() + I[2]*v.z();
            float Iv_y = I[1]*v.x() + I[3]*v.y() + I[4]*v.z();
            float Iv_z = I[2]*v.x() + I[4]*v.y() + I[5]*v.z();
            return u.x()*Iv_x + u.y()*Iv_y + u.z()*Iv_z;
        };
        float a = quadform(e1, e1);
        float b = quadform(e1, e2);
        float c = quadform(e2, e2);

        // --- Analytical eigendecomposition of symmetric 2×2 [[a,b],[b,c]] ---
        float mean  = (a + c) * 0.5f;
        float half  = (a - c) * 0.5f;
        float delta = std::sqrt(half * half + b * b);
        lam1 = mean + delta;   // dominant eigenvalue
        lam2 = mean - delta;   // weak eigenvalue (lam2 <= lam1)

        // --- Weak eigenvector in world frame ---
        // Eigenvector for lam2: [b, lam2-a] in the (e1,e2) basis.
        // Special case: I_2d ≈ 0 → completely unobserved, any direction is weak.
        float vx, vy;
        if (std::fabs(a - c) < 1e-8f && std::fabs(b) < 1e-8f) {
            vx = 1.0f; vy = 0.0f;
        } else {
            vx = b;
            vy = lam2 - a;
            float vn = std::sqrt(vx*vx + vy*vy);
            if (vn < 1e-8f) { vx = 1.0f; vy = 0.0f; }
            else { vx /= vn; vy /= vn; }
        }
        v_weak = e1 * vx + e2 * vy;
    }

    void Occupancy::get_2d_eigenstruct(const point3f &los,
                                       float &lam1, float &lam2,
                                       point3f &v_weak) const
    {
        compute_2d_eigenstruct_impl(info, los, lam1, lam2, v_weak);
    }

    std::ofstream &operator<<(std::ofstream &os, const Occupancy &oc) {
        os.write((char *) &oc.m_A, sizeof(oc.m_A));
        os.write((char *) &oc.m_B, sizeof(oc.m_B));
        return os;
    }

    std::ifstream &operator>>(std::ifstream &is, Occupancy &oc) {
        float m_A, m_B;
        is.read((char *) &m_A, sizeof(m_A));
        is.read((char *) &m_B, sizeof(m_B));
        oc = OcTreeNode(m_A, m_B);
        return is;
    }

    std::ostream &operator<<(std::ostream &os, const Occupancy &oc) {
        return os << '(' << oc.m_A << ' ' << oc.m_B << ' ' << oc.get_prob() << ')';
    }
}
