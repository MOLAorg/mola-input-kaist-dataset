/* -------------------------------------------------------------------------
 *   A Modular Optimization framework for Localization and mApping  (MOLA)
 * Copyright (C) 2018-2021 Jose Luis Blanco, University of Almeria
 * See LICENSE for license information.
 * ------------------------------------------------------------------------- */

/**
 * @file   KaistDataset.cpp
 * @brief  RawDataSource from the KAIST urban dataset
 * @author Jose Luis Blanco Claraco
 * @date   May 9, 2019
 */

/** \defgroup mola_input_kaist_dataset_grp mola_input_kaist_dataset_grp.
 * RawDataSource from the KAIST urban dataset
 */

#include <mola-input-kaist-dataset/KaistDataset.h>
#include <mola-kernel/variant_helper.h>
#include <mola-yaml/yaml_helpers.h>
#include <mrpt/core/initializer.h>
#include <mrpt/io/vector_loadsave.h>
#include <mrpt/obs/CActionRobotMovement2D.h>
#include <mrpt/obs/CObservationIMU.h>
#include <mrpt/obs/CObservationImage.h>
#include <mrpt/obs/CObservationOdometry.h>
#include <mrpt/obs/CObservationPointCloud.h>
#include <mrpt/system/filesystem.h>  //ASSERT_DIRECTORY_EXISTS_()
#include <mrpt/containers/yaml.h>
#include <Eigen/Dense>
// Eigen must be before csv.h
#include <mrpt/io/csv.h>

using namespace mola;

// arguments: class_name, parent_class, class namespace
IMPLEMENTS_MRPT_OBJECT(KaistDataset, RawDataSourceBase, mola)

MRPT_INITIALIZER(do_register_KaistDataset)
{
    MOLA_REGISTER_MODULE(KaistDataset);
}

KaistDataset::KaistDataset() = default;

void KaistDataset::initialize(const std::string& cfg_block)
{
    using namespace std::string_literals;

    MRPT_START
    ProfilerEntry tle(profiler_, "initialize");

    // Mandatory parameters:
    auto c = mrpt::containers::yaml::FromText(cfg_block);

    ENSURE_YAML_ENTRY_EXISTS(c, "params");
    auto cfg = c["params"];
    MRPT_LOG_DEBUG_STREAM("Initializing with these params:\n" << cfg);

    YAML_LOAD_MEMBER_REQ(base_dir, std::string);
    YAML_LOAD_MEMBER_REQ(sequence, std::string);

    // Optional params with default values:
    YAML_LOAD_MEMBER_OPT(time_warp_scale, double);

    // Preload everything we may need later to quickly replay the dataset in
    // realtime:
    seq_dir_ = base_dir_ + "/"s + sequence_ + "/sensor_data"s;
    ASSERT_DIRECTORY_EXISTS_(seq_dir_);

    // ======== CALIBRATION ============
    const auto calib_dir = base_dir_ + "/"s + sequence_ + "/calibration"s;
    ASSERT_DIRECTORY_EXISTS_(calib_dir);

    MRPT_LOG_INFO_STREAM("Loading KAIST dataset from: " << seq_dir_);

    // Sensor poses on the vehicle:
    const auto calib_vlp_l = calib_dir + "/Vehicle2LeftVLP.txt"s;
    auto&      pose_vlp_l  = calib_.pose_VLP[IDX_VLP_LEFT];  // shortcut
    pose_vlp_l             = load_pose_from_kaist_txt(calib_vlp_l);

    const auto calib_vlp_r = calib_dir + "/Vehicle2RightVLP.txt"s;
    auto&      pose_vlp_r  = calib_.pose_VLP[IDX_VLP_RIGHT];  // shortcut
    pose_vlp_r             = load_pose_from_kaist_txt(calib_vlp_r);

    MRPT_LOG_DEBUG_STREAM(
        "Left VLP pose: "
        << pose_vlp_l.asString() << " = "
        << pose_vlp_l.getHomogeneousMatrix().inMatlabFormat());
    MRPT_LOG_DEBUG_STREAM(
        "Right VLP pose: "
        << pose_vlp_r.asString() << " = "
        << pose_vlp_r.getHomogeneousMatrix().inMatlabFormat());

    // Wheels odometry calibration:
    const auto calib_odom = calib_dir + "/EncoderParameter.txt"s;
    ASSERT_FILE_EXISTS_(calib_odom);
    load_encoder_calib_file(calib_odom);

    // ======== DATA ============
    const auto data_dir = base_dir_ + "/"s + sequence_ + "/sensor_data/"s;
    ASSERT_DIRECTORY_EXISTS_(data_dir);

    const std::array<std::string, 2> VLP_prefix = {"VLP_left"s, "VLP_right"s};

    // Velodynes: {0,1}
    for (uint8_t vlp_id = 0; vlp_id < VLP_COUNT; vlp_id++)
    {
        const auto tim_data_fil = data_dir + VLP_prefix[vlp_id] + "_stamp.csv"s;
        ASSERT_FILE_EXISTS_(tim_data_fil);

        Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> timestamps;

        MRPT_LOG_DEBUG_STREAM("Loading: " << tim_data_fil);

        mrpt::io::load_csv(tim_data_fil, timestamps);

        MRPT_LOG_DEBUG("Done.");
        ASSERT_(timestamps.cols() == 1);
        ASSERT_(timestamps.rows() > 10);

        for (int row = 0; row < timestamps.rows(); row++)
        {
            const auto t = static_cast<kaist_timestamp_t>(timestamps(row, 0));
            const auto vlp_filename = data_dir + VLP_prefix[vlp_id] + "/"s +
                                      std::to_string(t) + ".bin"s;
            ASSERT_FILE_EXISTS_(vlp_filename);

            auto o = mrpt::obs::CObservationPointCloud::Create();

            o->timestamp   = mrpt::Clock::fromDouble(t * KAIST2UNIXTIME_FACTOR);
            o->sensorLabel = VLP_prefix[vlp_id];
            o->setAsExternalStorage(
                vlp_filename, mrpt::obs::CObservationPointCloud::
                                  ExternalStorageFormat::KittiBinFile);

            o->sensorPose = mrpt::poses::CPose3D(calib_.pose_VLP[vlp_id]);

            // Append to dataset:
            SensorVelodyne se_vlp;
            se_vlp.sensor_name = o->sensorLabel;
            se_vlp.obs         = o;
            dataset_.emplace_hint(dataset_.end(), t, se_vlp);
        }
        MRPT_LOG_DEBUG_STREAM(
            "Done reading "
            << timestamps.rows()
            << " Velodyne scans for sensor: " << VLP_prefix[vlp_id]);
    }

    // Odometry (wheels encoders):
    {
        const auto odo_data_fil = data_dir + "encoder.csv"s;
        ASSERT_FILE_EXISTS_(odo_data_fil);

        Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> odo;

        MRPT_LOG_DEBUG_STREAM("Loading: " << odo_data_fil);

        mrpt::io::load_csv(odo_data_fil, odo);

        MRPT_LOG_DEBUG("Done.");
        ASSERT_(odo.cols() == 3);
        ASSERT_(odo.rows() > 10);

        // Aux object to compute the wheels odometry kinematic equation:
        mrpt::obs::CActionRobotMovement2D odoIncr;
        odoIncr.hasEncodersInfo = true;
        // meters/tick ratios:
        const double K_left =
            (calib_.odom_left_diameter * M_PI) / calib_.odom_encoder_res;
        const double K_right =
            (calib_.odom_right_diameter * M_PI) / calib_.odom_encoder_res;

        mrpt::poses::CPose2D accum_odo(0, 0, 0);
        constexpr auto INVALID_ENC_COUNT = std::numeric_limits<int32_t>::max();

        int32_t last_encoder_l = INVALID_ENC_COUNT,
                last_encoder_r = INVALID_ENC_COUNT;

        for (int row = 0; row < odo.rows(); row++)
        {
            const auto t         = static_cast<kaist_timestamp_t>(odo(row, 0));
            const auto encoder_l = static_cast<int32_t>(odo(row, 1));
            const auto encoder_r = static_cast<int32_t>(odo(row, 2));

            // Increment odom:
            if (last_encoder_l != INVALID_ENC_COUNT)
            {
                odoIncr.encoderLeftTicks  = encoder_l - last_encoder_l;
                odoIncr.encoderRightTicks = encoder_r - last_encoder_r;
            }
            last_encoder_l = encoder_l;
            last_encoder_r = encoder_r;

            odoIncr.computeFromEncoders(
                K_left, K_right, calib_.odom_wheel_base);

            // and accumulate:
            accum_odo = accum_odo + odoIncr.rawOdometryIncrementReading;

#if 0  // **too** verbose. Left here for debugging and to help understanding.
            MRPT_LOG_DEBUG_STREAM(
                "odo incr: Al=" << encoder_l << " Ar=" << encoder_r
                                << " incrPose="
                                << odoIncr.rawOdometryIncrementReading
                                << " accumOdom=" << accum_odo);
#endif

            // Create observation object:
            auto o = mrpt::obs::CObservationOdometry::Create();

            o->timestamp   = mrpt::Clock::fromDouble(t * KAIST2UNIXTIME_FACTOR);
            o->sensorLabel = "wheels_odometry";
            o->odometry    = accum_odo;
            o->hasEncodersInfo   = true;
            o->encoderLeftTicks  = encoder_l;
            o->encoderRightTicks = encoder_r;

            // Append to dataset:
            SensorOdometry se_odo;
            se_odo.sensor_name = o->sensorLabel;
            se_odo.obs         = o;
            dataset_.emplace_hint(dataset_.end(), t, se_odo);
        }
        MRPT_LOG_DEBUG_STREAM(
            "Done reading " << odo.rows() << " wheels odometry entries.");
    }

    // Start at the dataset begin:
    dataset_next_ = dataset_.begin();

    MRPT_LOG_INFO_STREAM(
        "KAIST dataset preload done. Number of dataset entries: "
        << dataset_.size());

    MRPT_END
}  // end initialize()

void KaistDataset::spinOnce()
{
    MRPT_START

    ProfilerEntry tleg(profiler_, "spinOnce");

    // Starting time:
    if (!replay_started_)
    {
        replay_begin_time_ = mrpt::Clock::now();
        replay_started_    = true;
    }

    // get current replay time:
    const double t =
        mrpt::system::timeDifference(replay_begin_time_, mrpt::Clock::now()) *
        time_warp_scale_;

    if (dataset_next_ == dataset_.end())
    {
        MRPT_LOG_THROTTLE_INFO(
            10.0,
            "End of dataset reached! Nothing else to publish (CTRL+C to quit)");
        return;
    }

    // We have to publish all observations until "t":
    while (dataset_next_ != dataset_.end() &&
           t + (dataset_.begin()->first * KAIST2UNIXTIME_FACTOR) >=
               dataset_next_->first * KAIST2UNIXTIME_FACTOR)
    {
        // Convert from dataset format:
        const auto obs_tim = mrpt::Clock::fromDouble(
            dataset_next_->first * KAIST2UNIXTIME_FACTOR);

        std::visit(
            overloaded{[&](std::monostate&) {
                           THROW_EXCEPTION("Un-initialized entry!");
                       },
                       [&](SensorOdometry& odo) {
                           build_dataset_entry_obs(odo);
                           odo.obs->timestamp = obs_tim;
                           this->sendObservationsToFrontEnds(odo.obs);
                           odo.obs.reset();  // free mem
                       },
                       [&](SensorVelodyne& vlp) {
                           build_dataset_entry_obs(vlp);
                           vlp.obs->timestamp = obs_tim;
                           this->sendObservationsToFrontEnds(vlp.obs);
                           vlp.obs.reset();  // free mem
                       },
                       [&](SensorCamera& cam) {
                           build_dataset_entry_obs(cam);
                           cam.obs->timestamp = obs_tim;
                           this->sendObservationsToFrontEnds(cam.obs);
                           cam.obs.reset();  // free mem
                       },
                       [&](SensorSICK& sick) {
                           build_dataset_entry_obs(sick);
                           sick.obs->timestamp = obs_tim;
                           this->sendObservationsToFrontEnds(sick.obs);
                           sick.obs.reset();  // free mem
                       },
                       [&](SensorIMU& imu) {
                           build_dataset_entry_obs(imu);
                           imu.obs->timestamp = obs_tim;
                           this->sendObservationsToFrontEnds(imu.obs);
                           imu.obs.reset();  // free mem
                       }},
            dataset_next_->second);

        // Advance:
        ++dataset_next_;
    }

    // Read ahead to save delays in the next iteration:
    {
        ProfilerEntry      tle(profiler_, "spinOnce.read_ahead");
        const unsigned int READ_AHEAD_COUNT = 15;
        auto               peeker           = dataset_next_;
        ++peeker;
        for (unsigned int i = 0;
             i < READ_AHEAD_COUNT && peeker != dataset_.end(); ++i, ++peeker)
        {
            //
            std::visit(
                overloaded{
                    [&](std::monostate&) {
                        THROW_EXCEPTION("Un-initialized entry!");
                    },
                    [&](SensorOdometry& odo) { build_dataset_entry_obs(odo); },
                    [&](SensorVelodyne& vlp) { build_dataset_entry_obs(vlp); },
                    [&](SensorCamera& cam) { build_dataset_entry_obs(cam); },
                    [&](SensorSICK& sick) { build_dataset_entry_obs(sick); },
                    [&](SensorIMU& imu) { build_dataset_entry_obs(imu); }},
                peeker->second);
        }
    }

    MRPT_END
}

mrpt::math::TPose3D mola::load_pose_from_kaist_txt(const std::string& filename)
{
    using namespace std::string_literals;

    MRPT_START
    ASSERT_FILE_EXISTS_(filename);

    std::vector<std::string> lines;
    if (!mrpt::io::loadTextFile(lines, filename))
        THROW_EXCEPTION_FMT("Error reading filename: `%s`", filename.c_str());

    // clang-format off
    /** Example content:
     *
     * Left Velodyne (VLP-16) extrinsic calibration parameter from vehicle
     * RPY(roll/pitch/yaw, degree), R(rotation matrix), T(translation matrix)
     * RPY: 1.69162 44.9466 136.579
     * R: -0.514066 -0.702201 -0.492595 0.486485 -0.711672 0.506809 -0.706447 0.0208933 0.707457
     * T: -0.440699 0.397052 1.90953*
     */
    // clang-format on

    ASSERT_(lines.size() >= 5);
    ASSERT_(lines[3].substr(0, 2) == "R:"s);
    ASSERT_(lines[4].substr(0, 2) == "T:"s);

    const auto rot_str   = "["s + lines[3].substr(2) + "]"s;
    const auto trans_str = "["s + lines[4].substr(2) + "]"s;

    mrpt::math::CMatrixDouble rot_mat;
    if (!rot_mat.fromMatlabStringFormat(rot_str))
        THROW_EXCEPTION_FMT(
            "Error parsing rotation matrix line: `%s`", lines[3].c_str());
    ASSERT_EQUAL_(rot_mat.rows(), 1UL);
    ASSERT_EQUAL_(rot_mat.cols(), 9UL);

    mrpt::math::CMatrixDouble trans_mat;
    if (!trans_mat.fromMatlabStringFormat(trans_str))
        THROW_EXCEPTION_FMT(
            "Error parsing translation line: `%s`", lines[4].c_str());
    ASSERT_EQUAL_(trans_mat.rows(), 1UL);
    ASSERT_EQUAL_(trans_mat.cols(), 3UL);

    mrpt::math::CMatrixDouble33 R;
    for (int i = 0; i < 9; i++) R(i) = rot_mat(0, i);

    mrpt::poses::CPose3D p;
    p.setRotationMatrix(R);
    p.x(trans_mat(0, 0));
    p.y(trans_mat(0, 1));
    p.z(trans_mat(0, 2));

    return p.asTPose();
    MRPT_END
}

void KaistDataset::build_dataset_entry_obs(SensorCamera& s)
{
    ProfilerEntry tleg(profiler_, "build_obs_img");

#if 0
    if (s.obs) return;  // already done

    auto obs         = mrpt::obs::CObservationImage::Create();
    obs->sensorLabel = s.sensor_name;

    const auto f = seq_dir_ + s.img_file_name;
    obs->image.setExternalStorage(f);

    // Use this thread time to load images from disk, instead of
    // delegating it to the first use of the image in the consumer:
    obs->image.forceLoad();

    obs->cameraParams = cam_intrinsics_[s.cam_idx];
    obs->setSensorPose(mrpt::poses::CPose3D(cam_poses_[s.cam_idx]));

    s.obs = mrpt::ptr_cast<mrpt::obs::CObservation>::from(obs);
#endif
}

void KaistDataset::build_dataset_entry_obs(SensorIMU& s)
{
    using namespace mrpt::obs;

    ProfilerEntry tleg(profiler_, "build_obs_imu");

#if 0
    if (s.obs) return;  // already done

    MRPT_TODO("Port to CObservationIMU::CreateAlloc() with mem pool");

    auto obs         = CObservationIMU::Create();
    obs->sensorLabel = s.sensor_name;

    obs->dataIsPresent[IMU_WX]    = true;
    obs->dataIsPresent[IMU_WY]    = true;
    obs->dataIsPresent[IMU_WZ]    = true;
    obs->dataIsPresent[IMU_X_ACC] = true;
    obs->dataIsPresent[IMU_Y_ACC] = true;
    obs->dataIsPresent[IMU_Z_ACC] = true;

    obs->rawMeasurements[IMU_WX]    = s.wx;
    obs->rawMeasurements[IMU_WY]    = s.wy;
    obs->rawMeasurements[IMU_X_ACC] = s.accx;
    obs->rawMeasurements[IMU_Y_ACC] = s.accy;
    obs->rawMeasurements[IMU_Z_ACC] = s.accz;

    s.obs = mrpt::ptr_cast<mrpt::obs::CObservation>::from(obs);
#endif
}

void KaistDataset::build_dataset_entry_obs(SensorVelodyne& s)
{
    using namespace mrpt::obs;

    ProfilerEntry tleg(profiler_, "build_obs_vlp");

    auto ptr_o = dynamic_cast<mrpt::obs::CObservationPointCloud*>(s.obs.get());
    ASSERT_(ptr_o != nullptr);

    if (ptr_o->pointcloud && ptr_o->pointcloud->size() > 0)
        return;  // already done

    // Force load from disk now:
    ptr_o->load();
    ASSERTMSG_(
        ptr_o->pointcloud, mrpt::format(
                               "Error loading scan file: '%s'",
                               ptr_o->getExternalStorageFile().c_str()));
}

void KaistDataset::build_dataset_entry_obs(SensorOdometry& s)
{
    using namespace mrpt::obs;

    ProfilerEntry tleg(profiler_, "build_obs_odom");

#if 0
    if (s.obs) return;  // already done

    MRPT_TODO("Port to CObservationIMU::CreateAlloc() with mem pool");

    auto obs         = CObservationIMU::Create();
    obs->sensorLabel = s.sensor_name;

    obs->dataIsPresent[IMU_WX]    = true;
    obs->dataIsPresent[IMU_WY]    = true;
    obs->dataIsPresent[IMU_WZ]    = true;
    obs->dataIsPresent[IMU_X_ACC] = true;
    obs->dataIsPresent[IMU_Y_ACC] = true;
    obs->dataIsPresent[IMU_Z_ACC] = true;

    obs->rawMeasurements[IMU_WX]    = s.wx;
    obs->rawMeasurements[IMU_WY]    = s.wy;
    obs->rawMeasurements[IMU_X_ACC] = s.accx;
    obs->rawMeasurements[IMU_Y_ACC] = s.accy;
    obs->rawMeasurements[IMU_Z_ACC] = s.accz;

    s.obs = mrpt::ptr_cast<mrpt::obs::CObservation>::from(obs);
#endif
}

void KaistDataset::build_dataset_entry_obs(SensorSICK& s)
{
    using namespace mrpt::obs;

    ProfilerEntry tleg(profiler_, "build_obs_sick");

#if 0
    if (s.obs) return;  // already done

    MRPT_TODO("Port to CObservationIMU::CreateAlloc() with mem pool");

    auto obs         = CObservationIMU::Create();
    obs->sensorLabel = s.sensor_name;

    obs->dataIsPresent[IMU_WX]    = true;
    obs->dataIsPresent[IMU_WY]    = true;
    obs->dataIsPresent[IMU_WZ]    = true;
    obs->dataIsPresent[IMU_X_ACC] = true;
    obs->dataIsPresent[IMU_Y_ACC] = true;
    obs->dataIsPresent[IMU_Z_ACC] = true;

    obs->rawMeasurements[IMU_WX]    = s.wx;
    obs->rawMeasurements[IMU_WY]    = s.wy;
    obs->rawMeasurements[IMU_X_ACC] = s.accx;
    obs->rawMeasurements[IMU_Y_ACC] = s.accy;
    obs->rawMeasurements[IMU_Z_ACC] = s.accz;

    s.obs = mrpt::ptr_cast<mrpt::obs::CObservation>::from(obs);
#endif
}

void KaistDataset::load_encoder_calib_file(const std::string& filename)
{
    using namespace std::string_literals;

    MRPT_START
    ASSERT_FILE_EXISTS_(filename);

    std::vector<std::string> lines;
    if (!mrpt::io::loadTextFile(lines, filename))
        THROW_EXCEPTION_FMT("Error reading filename: `%s`", filename.c_str());

    // clang-format off
    /** Example content:
     *
     * 0|Encoder calibrated parameter
     * 1|Encoder resolution: 4096
     * 2|Encoder left wheel diameter: 0.623022
     * 3|Encoder right wheel diameter: 0.622356
     * 4|Encoder wheel base: 1.5285
     *   0         1         2
     *   012345678901234567890123456789
     *
     */
    // clang-format on

    ASSERT_(lines.size() >= 5);
    ASSERT_EQUAL_(lines[1].substr(0, 19), "Encoder resolution:"s);
    calib_.odom_encoder_res = std::stoi(lines[1].substr(19));

    ASSERT_EQUAL_(lines[2].substr(0, 28), "Encoder left wheel diameter:"s);
    calib_.odom_left_diameter = std::stod(lines[2].substr(28));

    ASSERT_EQUAL_(lines[3].substr(0, 29), "Encoder right wheel diameter:"s);
    calib_.odom_right_diameter = std::stod(lines[3].substr(29));

    ASSERT_EQUAL_(lines[4].substr(0, 19), "Encoder wheel base:"s);
    calib_.odom_wheel_base = std::stod(lines[4].substr(19));

    MRPT_LOG_DEBUG_STREAM("odom_encoder_res:" << calib_.odom_encoder_res);
    MRPT_LOG_DEBUG_STREAM("odom_left_diameter:" << calib_.odom_left_diameter);
    MRPT_LOG_DEBUG_STREAM("odom_right_diameter:" << calib_.odom_right_diameter);
    MRPT_LOG_DEBUG_STREAM("odom_wheel_base:" << calib_.odom_wheel_base);

    MRPT_END
}
