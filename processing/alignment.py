import subprocess
import nibabel as nib
import numpy as np
from scipy.io import loadmat


def run(project):
    #PATSH

    settings = project.settings.processing
    plot = settings.alignment_plot

    project.output_dir.mkdir(parents=True, exist_ok=True)

    out_dir = project.anatomy_dir / "voxel_volumes"
    out_dir.mkdir(parents=True, exist_ok=True)

    coords_path = project.input_dir / "coords.mat"
    t1_path = project.input_dir / "t1_1mm.nii"
    wm_path = project.data_dir / "s1" / "mri" / "wm.mgz"

    nii_path = out_dir / "sparse_voxel_volume.nii.gz"
    mgz_path = out_dir / "sparse_voxel_volume.mgz"
    out_wm = out_dir / "voxels_in_fs.mgz"

    # Load anatomical images and mrVista coordinates.
    t1 = nib.load(t1_path)
    wm_img = nib.load(wm_path)

    shape = t1.shape

    coords = loadmat(coords_path)["coords"].T
    coords -= 1

    # Convert mrVista voxel indices to T1 voxel space.
    coords = coords[:, (2, 1, 0)]
    coords[:, 1] = shape[1] - 1 - coords[:, 1]
    coords[:, 2] = shape[2] - 1 - coords[:, 2]

    # Convert to homogeneous coordinates for affine transformations.
    coords_h = np.c_[coords, np.ones(coords.shape[0])]

    # Transform T1 voxel coordinates to scanner RAS.
    ras = (t1.affine @ coords_h.T).T

    # Transform scanner RAS to FreeSurfer white matter voxel space.
    inv_wm_affine = np.linalg.inv(wm_img.affine)
    coords_wm = (inv_wm_affine @ ras.T).T

    # Retrieve the FreeSurfer voxel-to-tkRAS transformation.
    result = subprocess.run(
        [
            "mri_info",
            "--vox2ras-tkr",
            str(project.data_dir / "s1" / "mri" / "orig.mgz"),
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )

    lines = result.stdout.strip().splitlines()
    torig = np.array([[float(x) for x in line.split()] for line in lines])

    # Convert white matter voxel coordinates to tkRAS.
    tkr_ras = (torig @ coords_wm.T).T[:, :3]

    out_path = project.input_dir / "coords_tkrRAS.npz"

    # Save transformed coordinates for subsequent processing.
    np.savez(out_path,tkrRAS=tkr_ras,mcoords_wm=coords_wm[:, :3],coords_t1=coords)

    print(f"Saved: {out_path}")

    if not plot:
        return

    # Create a sparse volume for visual inspection of the alignment.
    data = np.zeros(shape, dtype=np.uint8)
    data[coords[:, 0], coords[:, 1], coords[:, 2],] = 1

    img = nib.Nifti1Image(data, t1.affine, t1.header)
    nib.save(img, nii_path)

    print(f"Saved: {nii_path.name} | voxels: {len(coords)}")

    run_command(
        [
            "mri_convert",
            str(nii_path),
            str(mgz_path),
        ],
        "Converting sparse volume to MGZ",
    )

    run_command(
        [
            "mri_vol2vol",
            "--mov",
            str(mgz_path),
            "--targ",
            str(wm_path),
            "--regheader",
            "--nearest",
            "--o",
            str(out_wm),
        ],
        "Mapping sparse voxels into white matter space",
    )

    run_command(
        [
            "freeview",
            "-v",
            str(project.data_dir / "s1" / "mri" / "brain.mgz"),
            "-v",
            f"{out_wm}:colormap=heat:opacity=0.4",
            "-f",
            str(project.surf_dir / "lh.white"),
            "-f",
            str(project.surf_dir / "rh.white"),
        ],
        "Launching Freeview",
    )


def run_command(command, description):

    print(f"\n>>> {description}")
    print(" ".join(command))
    subprocess.run(command, check=True)