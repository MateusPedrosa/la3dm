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
    float Occupancy::tau_var  = 0.01f;
    float Occupancy::tau_info = 0.5f;

    Occupancy::Occupancy(float A, float B)
        : m_A(Occupancy::prior_A + A), m_B(Occupancy::prior_B + B),
          classified(false), is_trusted(false),
          lambda_max_cache(0.0f), info_constrained(false) {
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

    // Compute the minimum eigenvalue of a 3x3 symmetric matrix stored as upper
    // triangle [Ixx, Ixy, Ixz, Iyy, Iyz, Izz].
    //
    // Uses the analytical algorithm from Smith (1961) as described at:
    // https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3%C3%973_matrices
    //
    // Only called when both fast pre-filters pass (lambda_max_cache and variance),
    // so it does not need to be on the hot path.
    static float compute_min_eigenvalue_3x3(const float I[6]) {
        // Matrix: [[a,b,c],[b,d,e],[c,e,f]]
        const float a = I[0], b = I[1], c = I[2];
        const float d = I[3], e = I[4];
        const float f = I[5];

        float p1 = b*b + c*c + e*e;  // sum of squared off-diagonal elements

        if (p1 < 1e-12f) {
            // Diagonal matrix; eigenvalues are the diagonal elements
            return std::min({a, d, f});
        }

        float q = (a + d + f) / 3.0f;  // mean eigenvalue
        float p2 = (a-q)*(a-q) + (d-q)*(d-q) + (f-q)*(f-q) + 2.0f * p1;
        float p  = std::sqrt(p2 / 6.0f);

        if (p < 1e-12f) return q;  // degenerate: all eigenvalues equal q

        // B = (1/p) * (A - q*I)
        float Bxx = (a - q) / p, Bxy = b / p, Bxz = c / p;
        float Byy = (d - q) / p, Byz = e / p;
        float Bzz = (f - q) / p;

        // r = det(B) / 2
        float r = (Bxx*(Byy*Bzz - Byz*Byz)
                 - Bxy*(Bxy*Bzz - Byz*Bxz)
                 + Bxz*(Bxy*Byz - Byy*Bxz)) * 0.5f;

        // Clamp r to [-1, 1] to guard against floating-point drift
        if (r <= -1.0f) r = -1.0f;
        else if (r >= 1.0f) r = 1.0f;

        float phi = std::acos(r) / 3.0f;

        // Three eigenvalues (descending order)
        float eig1 = q + 2.0f * p * std::cos(phi);
        float eig3 = q + 2.0f * p * std::cos(phi + 2.0f * static_cast<float>(M_PI) / 3.0f);
        float eig2 = 3.0f * q - eig1 - eig3;

        return std::min({eig1, eig2, eig3});
    }

    void Occupancy::check_deallocation() {
        if (info_constrained) return;

        // Fast pre-filter 1: if the mean eigenvalue (trace/3) is below tau_info,
        // then lambda_min < tau_info for certain (lambda_min <= mean eigenvalue).
        if (lambda_max_cache < Occupancy::tau_info * 3.0f) return;

        // Fast pre-filter 2: sufficient total evidence (low Beta variance) required.
        if (get_var() >= Occupancy::tau_var) return;

        // Full check: compute exact minimum eigenvalue.
        float lambda_min = compute_min_eigenvalue_3x3(info);
        if (lambda_min > Occupancy::tau_info) {
            info_constrained = true;
        }
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
