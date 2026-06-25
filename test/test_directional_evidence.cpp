#include <gtest/gtest.h>
#include <cmath>
#include "bgksoctree_node.h"

// Helper: compute n^T I(p) n from an Occupancy's raw info.
static float quad_form(const la3dm::Occupancy& node, const la3dm::point3f& n)
{
    float info[6];
    node.get_info(info);
    float x = n.x(), y = n.y(), z = n.z();
    return info[0]*x*x + info[3]*y*y + info[5]*z*z
         + 2.0f*info[1]*x*y + 2.0f*info[2]*x*z + 2.0f*info[4]*y*z;
}

// Feed `count` identical observations from direction n with weight w_range.
// Simulates the gated update sequence: w_voxel computed pre-update,
// then w_total = w_range * w_voxel applied to the info matrix.
static void feed_obs(la3dm::Occupancy& node, const la3dm::point3f& n,
                     float w_range, int count)
{
    const la3dm::point3f los_hat(0.0f, 0.0f, -1.0f);  // arbitrary; not used by check_deallocation in these tests (tau_var high)
    for (int i = 0; i < count; ++i) {
        float w_voxel = node.compute_w_novelty(n);
        float w_total = w_range * w_voxel;
        node.update_info_matrix(n, w_total);
        // check_deallocation is skipped here (tau_var = 1000 disables it)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Asymptotic saturation — n1^T I(p) n1 never exceeds W_sat,
//         approaches it after 50 observations, and w_voxel(n1) is
//         monotonically non-increasing.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DirectionalEvidence, AsymptoticSaturation)
{
    la3dm::Occupancy::tau_info = 1.0f;
    la3dm::Occupancy::tau_var  = 1000.0f;
    la3dm::Occupancy::delta    = 0.05f;

    const float W_sat   = la3dm::Occupancy::tau_info;
    const float w_range = 0.1f;
    const la3dm::point3f n1(1.0f, 0.0f, 0.0f);

    la3dm::Occupancy node;
    float prev_w_voxel = 1.0f;

    for (int k = 0; k < 50; ++k) {
        float w_voxel = node.compute_w_novelty(n1);

        // Monotonically non-increasing
        EXPECT_LE(w_voxel, prev_w_voxel + 1e-6f)
            << "w_voxel increased at observation " << k;
        prev_w_voxel = w_voxel;

        float w_total = w_range * w_voxel;
        node.update_info_matrix(n1, w_total);

        float nIn = quad_form(node, n1);
        EXPECT_LE(nIn, W_sat + 1e-4f)
            << "n1^T I n1 exceeded W_sat at observation " << k;
    }

    // After 50 observations, must have approached W_sat
    float nIn_final = quad_form(node, n1);
    EXPECT_GT(nIn_final, 0.9f * W_sat)
        << "n1^T I n1 did not approach W_sat after 50 observations";
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: Leakage fix — w_voxel(n2) after saturation at n1 is bounded below
//         by approximately sin^2(theta), and does not depend on whether 50
//         or 500 prior redundant observations were made at n1.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DirectionalEvidence, BoundedCrossDirectionEffect)
{
    la3dm::Occupancy::tau_info = 1.0f;
    la3dm::Occupancy::tau_var  = 1000.0f;
    la3dm::Occupancy::delta    = 0.05f;

    const float W_sat   = la3dm::Occupancy::tau_info;
    const float w_range = 0.1f;
    const float theta_deg = 10.0f;
    const float theta_rad = theta_deg * M_PI / 180.0f;

    const la3dm::point3f n1(1.0f, 0.0f, 0.0f);
    const la3dm::point3f n2(std::cos(theta_rad), std::sin(theta_rad), 0.0f);

    const float cos2_theta = std::cos(theta_rad) * std::cos(theta_rad);
    const float sin2_theta = 1.0f - cos2_theta;

    // Case A: 50 prior observations at n1 (near saturation)
    la3dm::Occupancy node_50;
    feed_obs(node_50, n1, w_range, 50);
    float w_n2_50 = node_50.compute_w_novelty(n2);

    // Case B: 500 prior observations at n1 (deep saturation)
    la3dm::Occupancy node_500;
    feed_obs(node_500, n1, w_range, 500);
    float w_n2_500 = node_500.compute_w_novelty(n2);

    // Both should be close to sin^2(theta) once saturated
    EXPECT_NEAR(w_n2_50, w_n2_500, 0.005f)
        << "w_voxel(n2) differs between 50 and 500 prior n1 observations: "
        << w_n2_50 << " vs " << w_n2_500;

    // Bounded below by the angular formula: w_voxel >= 1 - W*cos^2(theta)/W_sat
    // At saturation W -> W_sat, so lower bound is sin^2(theta).
    // Allow 10% tolerance for asymptotic approach.
    EXPECT_GE(w_n2_500, 0.9f * sin2_theta)
        << "w_voxel(n2) after 500 obs is below the angular lower bound";

    // Crucially: must be strictly positive (not suppressed to zero).
    EXPECT_GT(w_n2_500, 0.0f)
        << "w_voxel(n2) was suppressed to zero by redundant n1 observations";

    // Cross-check: value is close to sin^2(theta) from above as well
    // (W <= W_sat by saturation, so n2^T I n2 = W*cos^2 <= W_sat*cos^2)
    EXPECT_LE(w_n2_500, sin2_theta + 0.02f)
        << "w_voxel(n2) is above the saturation ceiling";

    (void)W_sat;  // used in comment above
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: Exact orthogonal immunity — w_voxel for a direction exactly
//         orthogonal to all past observations is always exactly 1.0.
// ─────────────────────────────────────────────────────────────────────────────
TEST(DirectionalEvidence, OrthogonalImmunity)
{
    la3dm::Occupancy::tau_info = 1.0f;
    la3dm::Occupancy::tau_var  = 1000.0f;
    la3dm::Occupancy::delta    = 0.05f;

    const float w_range = 0.1f;
    const la3dm::point3f n1(1.0f, 0.0f, 0.0f);
    const la3dm::point3f n3(0.0f, 1.0f, 0.0f);  // exactly orthogonal to n1

    // Before any observations: fresh voxel
    la3dm::Occupancy node_fresh;
    EXPECT_FLOAT_EQ(node_fresh.compute_w_novelty(n3), 1.0f)
        << "w_voxel(n3) is not 1 for a fresh voxel";

    // After 50 saturating observations from n1
    la3dm::Occupancy node_sat;
    feed_obs(node_sat, n1, w_range, 50);

    float w_n3 = node_sat.compute_w_novelty(n3);
    EXPECT_NEAR(w_n3, 1.0f, 1e-4f)
        << "w_voxel(n3 ⊥ n1) is not 1 after 50 observations at n1: " << w_n3;

    // After 500 observations
    la3dm::Occupancy node_deep;
    feed_obs(node_deep, n1, w_range, 500);

    float w_n3_deep = node_deep.compute_w_novelty(n3);
    EXPECT_NEAR(w_n3_deep, 1.0f, 1e-4f)
        << "w_voxel(n3 ⊥ n1) is not 1 after 500 observations at n1: " << w_n3_deep;
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
