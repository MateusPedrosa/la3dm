#include "bgkloctree_node.h"
#include <cmath>

namespace la3dm {

    /// Default static values
    float Occupancy::sf2 = 1.0f;
    float Occupancy::ell = 1.0f;
    float Occupancy::free_thresh = 0.3f;
    float Occupancy::occupied_thresh = 0.7f;
    float Occupancy::var_thresh = 1000.0f;
    float Occupancy::prior_A = 0.5f;
    float Occupancy::prior_B = 0.5f;

    Occupancy::Occupancy(float A, float B) : m_A(Occupancy::prior_A + A), m_B(Occupancy::prior_B + B) {
        classified = false;
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

        const float R = 4.0f;

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