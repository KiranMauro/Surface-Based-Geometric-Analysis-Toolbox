import nibabel as nib
import numpy as np
import scipy.sparse

def run(project, smoothing_iterations=0):
    # SETTINGS
    model_type = project.model
    settings = project.settings.processing
    r2_threshold = settings.r2_threshold
    r2_display_threshold = settings.r2_display_threshold
    num_range = settings.num_range
    max_eccentricity = settings.max_eccentricity

    # PATHS
    parameter_map_dir = project.parameter_map_dir
    parameter_map_dir.mkdir(parents=True, exist_ok=True)

    # LOAD DATA
    vertex_data = project.vertex_data
    r2 = vertex_data.metrics["r2"]

    n_lh = len(project.geo_lh.verts)
    n_rh = len(project.geo_rh.verts)

    
    # PARAMETER SELECTION
   
    if model_type == "N":

        parameter_name = "N"

        parameter_values = vertex_data.metrics["x0"]

        log_num_range = np.log(num_range)

    elif model_type == "VFM":

        parameter_name = "polar_angle"

        parameter_values = vertex_data.metrics["polar_angle"]

        eccentricity = vertex_data.metrics["eccentricity"]

    else:

        raise ValueError(f"Unknown MODEL_TYPE: {model_type}")

   
    parameter_map = parameter_values.copy()

    if model_type == "N":

        parameter_mask = (
            (parameter_values >= log_num_range[0])
            &
            (parameter_values <= log_num_range[1])
        )

    else:

        parameter_mask = (
            eccentricity <= max_eccentricity
        )

    mask = (
        parameter_mask
        &
        (r2 >= r2_threshold)
    )


    parameter_map[~mask] = np.nan

    # SPLIT HEMISPHERES
    lh_parameter = parameter_map[:n_lh]
    rh_parameter = parameter_map[n_lh: n_lh + n_rh]

    
    lh_r2 = r2[:n_lh]
    rh_r2 = r2[n_lh:n_lh + n_rh]

    lh_r2_display = lh_r2.copy()
    rh_r2_display = rh_r2.copy()

    lh_mask = parameter_mask[:n_lh]
    rh_mask = parameter_mask[n_lh:]

    lh_r2_display = lh_r2.copy()
    rh_r2_display = rh_r2.copy()

    lh_r2_display[~lh_mask] = np.nan
    rh_r2_display[~rh_mask] = np.nan

    lh_r2_display[lh_r2_display < r2_display_threshold] = np.nan
    rh_r2_display[rh_r2_display < r2_display_threshold] = np.nan

    # SAVE MGH FILES
    lh_path = (parameter_map_dir/ f"lh.{parameter_name}.mgh")

    rh_path = (parameter_map_dir/ f"rh.{parameter_name}.mgh")

    nib.save(nib.MGHImage(lh_parameter[:, None, None].astype(np.float32),np.eye(4)),lh_path)
    nib.save(nib.MGHImage(rh_parameter[:, None, None].astype(np.float32),np.eye(4)),rh_path)
    nib.save(nib.MGHImage(lh_r2[:, None, None].astype(np.float32),np.eye(4)),parameter_map_dir / f"lh.r2_{model_type}.mgh")
    nib.save(nib.MGHImage(rh_r2[:, None, None].astype(np.float32),np.eye(4)),parameter_map_dir / f"rh.r2_{model_type}.mgh")

    if model_type == 'N':

        lh_display = np.exp(lh_parameter)
        rh_display = np.exp(rh_parameter)

        nib.save(nib.MGHImage(
                lh_display[:, None, None].astype(
                    np.float32
                ),
                np.eye(4)
            ),
            parameter_map_dir / f"lh.{model_type}_display.mgh"
        )

        nib.save(
            nib.MGHImage(
                rh_display[:, None, None].astype(
                    np.float32
                ),
                np.eye(4)
            ),
            parameter_map_dir / f"rh.{model_type}_display.mgh"
        )


        nib.save(
            nib.MGHImage(
                lh_r2_display[:, None, None].astype(np.float32),
                np.eye(4)
            ),
            parameter_map_dir / "lh.r2_display.mgh"
        )

        nib.save(
            nib.MGHImage(
                rh_r2_display[:, None, None].astype(np.float32),
                np.eye(4)
            ),
            parameter_map_dir / "rh.r2_display.mgh"
        )

    
    # SUMMARY
    print(f"parameter: {parameter_name}")

    print(f"Valid vertices: "f"{np.sum(mask)} / {len(mask)}")
