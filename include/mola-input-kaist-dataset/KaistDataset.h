/* -------------------------------------------------------------------------
 *   A Modular Optimization framework for Localization and mApping  (MOLA)
 * Copyright (C) 2018-2021 Jose Luis Blanco, University of Almeria
 * See LICENSE for license information.
 * ------------------------------------------------------------------------- */
/**
 * @file   KaistDataset.h
 * @brief  RawDataSource from the KAIST urban dataset
 * @author Jose Luis Blanco Claraco
 * @date   May 9, 2019
 */
#pragma once

#include <mola-kernel/interfaces/RawDataSourceBase.h>
#include <mrpt/core/Clock.h>
#include <mrpt/img/TCamera.h>
#include <mrpt/math/TPose3D.h>
#include <array>
#include <map>
#include <string>
#include <variant>

namespace mola
{
/** \addtogroup mola_input_kaist_dataset_grp
 * @{ */

/** Loads a pose from a TXT in KAIST format. Throws on error. */
mrpt::math::TPose3D load_pose_from_kaist_txt(const std::string& filename);

struct SensorGeneric
{
    std::string                  sensor_name;
    mrpt::obs::CObservation::Ptr obs;
};

struct SensorCamera : public SensorGeneric
{
    std::string img_file_name;
    uint8_t     cam_idx;
};
struct SensorIMU : public SensorGeneric
{
    double wx, wy, wz, accx, accy, accz;
};
struct SensorVelodyne : public SensorGeneric
{
};
struct SensorSICK : public SensorGeneric
{
};
struct SensorOdometry : public SensorGeneric
{
};

using SensorEntry = std::variant<
    std::monostate, SensorCamera, SensorIMU, SensorVelodyne, SensorSICK,
    SensorOdometry>;
using kaist_timestamp_t = uint64_t;
using kaist_dataset_t   = std::multimap<kaist_timestamp_t, SensorEntry>;

/** RawDataSource from KAIST dataset.
 * Each "sequence" directory contains these sensor streams:
 * - xxx
 * - Ground truth poses
 *
 */
class KaistDataset : public RawDataSourceBase
{
    DEFINE_MRPT_OBJECT(KaistDataset, mola)

   public:
    KaistDataset();
    ~KaistDataset() override = default;

    // See docs in base class
    void initialize(const std::string& cfg_block) override;
    void spinOnce() override;

    /** KAIST timestamp is Unix time * 1e9 */
    constexpr static double KAIST2UNIXTIME_FACTOR = 1e-9;

   private:
    std::string             base_dir_;  //!< base dir for `xxx/xx/mav0/...`
    std::string             sequence_;  //!< e.g. `machine_hall/MH_01_easy`
    mrpt::Clock::time_point replay_begin_time_{};
    bool                    replay_started_{false};
    double                  time_warp_scale_{1.0};
    bool                    publish_VLP_left{true}, publish_VLP_right{true};
    bool                    publish_SICK_back{true}, publish_SICK_middle{true};
    bool                    publish_odometry{true};

    enum VLP_indices_t : uint8_t
    {
        IDX_VLP_LEFT = 0,
        IDX_VLP_RIGHT,
        // ---
        VLP_COUNT
    };

    enum SICK_indices_t : uint8_t
    {
        IDX_SICK_BACK = 0,
        IDX_SICK_MIDDLE,
        // ---
        SICK_COUNT
    };

    // All poses are wrt vehicle frame of reference
    // (refer to figures in KAIST paper)
    struct Calibration
    {
        // Velodynes:
        std::array<mrpt::math::TPose3D, 2> pose_VLP;

        // Odometry:
        int    odom_encoder_res{4096};  //!< resolution
        double odom_left_diameter{0.623}, odom_right_diameter{0.622};
        double odom_wheel_base{1.5285};
    };

    /** Dataset calibration parameters */
    Calibration calib_;

    void load_encoder_calib_file(const std::string& filename);

    std::string seq_dir_;  //!< selected "sensor_data" directory

    kaist_dataset_t           dataset_;  //!< dataset itself
    kaist_dataset_t::iterator dataset_next_;  //!< next item to publish

    void build_dataset_entry_obs(SensorCamera& s);
    void build_dataset_entry_obs(SensorIMU& s);
    void build_dataset_entry_obs(SensorSICK& s);
    void build_dataset_entry_obs(SensorVelodyne& s);
    void build_dataset_entry_obs(SensorOdometry& s);
};

/** @} */

}  // namespace mola
