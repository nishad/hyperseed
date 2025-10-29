#!/usr/bin/env python3
"""Main command-line interface for hyperseed.

This module provides the CLI entry point and commands for the hyperspectral
seed analysis pipeline.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from hyperseed import __version__
from hyperseed.config.settings import Settings
from hyperseed.core.calibration import ReflectanceCalibrator
from hyperseed.core.preprocessing import PreprocessingPipeline
from hyperseed.core.segmentation import SeedSegmenter
from hyperseed.core.extraction import SpectralExtractor


# Setup console for rich output
console = Console()


def setup_logging(verbose: bool, debug: bool) -> None:
    """Setup logging configuration.

    Args:
        verbose: Enable verbose output.
        debug: Enable debug output.
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


@click.group()
@click.version_option(version=__version__)
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output.')
@click.option('-d', '--debug', is_flag=True, help='Enable debug output.')
@click.pass_context
def main(ctx, verbose: bool, debug: bool):
    """Hyperseed - Professional hyperspectral seed analysis tool.

    A comprehensive command-line tool for analyzing hyperspectral imagery
    of plant seeds, extracting average spectral signatures with spatial
    information.

    Examples:

        # Analyze a single dataset
        hyperseed analyze path/to/data --output results.csv

        # Batch process multiple datasets
        hyperseed batch path/to/datasets --config config.yaml

        # Only perform segmentation
        hyperseed segment path/to/data --visualize
    """
    setup_logging(verbose, debug)
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['debug'] = debug


@main.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option(
    '-o', '--output',
    type=click.Path(path_type=Path),
    help='Output file path (CSV or HDF5).'
)
@click.option(
    '-c', '--config',
    type=click.Path(exists=True, path_type=Path),
    help='Configuration file path (YAML).'
)
@click.option(
    '--preprocess',
    type=click.Choice(['minimal', 'standard', 'advanced', 'none']),
    default='standard',
    help='Preprocessing preset to use.'
)
@click.option(
    '--segmentation',
    type=click.Choice(['threshold', 'watershed', 'connected', 'combined']),
    default='watershed',
    help='Segmentation algorithm to use.'
)
@click.option(
    '--min-pixels',
    type=int,
    default=200,
    help='Minimum seed size in pixels.'
)
@click.option(
    '--export-plots',
    is_flag=True,
    help='Export visualization plots.'
)
@click.option(
    '--export-mask',
    is_flag=True,
    help='Export segmentation mask.'
)
@click.option(
    '--no-outlier-removal',
    is_flag=True,
    help='Disable automatic outlier removal.'
)
@click.pass_context
def analyze(
    ctx,
    input_path: Path,
    output: Optional[Path],
    config: Optional[Path],
    preprocess: str,
    segmentation: str,
    min_pixels: int,
    export_plots: bool,
    export_mask: bool,
    no_outlier_removal: bool
):
    """Analyze hyperspectral seed imagery.

    Performs the complete analysis pipeline: calibration, preprocessing,
    segmentation, and spectral extraction.

    Example:
        hyperseed analyze dataset/sample --output results.csv --export-plots
    """
    console.print(f"[bold green]Hyperseed Analysis[/bold green]")
    console.print(f"Input: {input_path}")

    # Load configuration
    if config:
        console.print(f"Loading configuration from {config}")
        settings = Settings.load(config)
    else:
        settings = Settings()
        settings.preprocessing.method = preprocess
        settings.segmentation.algorithm = segmentation
        settings.segmentation.min_pixels = min_pixels

    # Default output path if not specified
    if output is None:
        output = Path(f"{input_path.name}_spectra.csv")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:

            # Step 1: Calibration
            task = progress.add_task("Calibrating data...", total=None)
            calibrator = ReflectanceCalibrator(
                clip_negative=settings.calibration.clip_negative,
                clip_max=settings.calibration.clip_max
            )
            calibrated_data, reader = calibrator.calibrate_from_directory(input_path)
            wavelengths = reader.get_wavelengths()
            progress.update(task, completed=True)

            console.print(
                f"✓ Calibrated: {calibrated_data.shape} "
                f"({calibrated_data.shape[2]} bands)"
            )

            # Step 2: Preprocessing
            if preprocess != 'none':
                task = progress.add_task("Preprocessing...", total=None)
                preprocessor = PreprocessingPipeline(settings.preprocessing)
                processed_data = preprocessor.fit_transform(calibrated_data)
                progress.update(task, completed=True)

                console.print(f"✓ Preprocessed with {preprocess} method")
            else:
                processed_data = calibrated_data

            # Step 3: Segmentation
            task = progress.add_task("Segmenting seeds...", total=None)
            segmenter = SeedSegmenter(settings.segmentation)
            mask, n_seeds = segmenter.segment(processed_data, validate=True)
            progress.update(task, completed=True)

            console.print(f"✓ Segmented: {n_seeds} seeds found")

            # Step 4: Spectral Extraction
            task = progress.add_task("Extracting spectra...", total=None)
            extractor = SpectralExtractor()
            results = extractor.extract(
                calibrated_data,  # Use original calibrated data
                mask,
                wavelengths=wavelengths,
                compute_stats=True
            )
            progress.update(task, completed=True)

            console.print(f"✓ Extracted: {results['n_seeds']} seed spectra")

            # Apply outlier removal if configured
            if settings.segmentation.remove_outliers and results['n_seeds'] > 0 and not no_outlier_removal:
                # Convert segmentation config to outlier config dict
                outlier_config = {
                    'remove_outliers': settings.segmentation.remove_outliers,
                    'outlier_min_area': settings.segmentation.outlier_min_area,
                    'outlier_max_area': settings.segmentation.outlier_max_area,
                    'outlier_iqr_lower': settings.segmentation.outlier_iqr_lower,
                    'outlier_iqr_upper': settings.segmentation.outlier_iqr_upper,
                    'outlier_eccentricity': settings.segmentation.outlier_eccentricity,
                    'outlier_solidity': settings.segmentation.outlier_solidity,
                    'use_shape_filtering': settings.segmentation.use_shape_filtering
                }

                n_before = results['n_seeds']
                _, _, removed = extractor.remove_outliers(outlier_config, verbose=False)
                n_after = len(extractor.seed_info)

                if len(removed) > 0:
                    console.print(f"✓ Removed {len(removed)} outlier seeds ({n_before} → {n_after})")
                    # Update results with filtered data
                    results['n_seeds'] = n_after
                    results['spectra'] = extractor.spectra
                    results['seed_info'] = extractor.seed_info

            # Step 5: Export results
            task = progress.add_task("Saving results...", total=None)

            # Save spectra only if seeds were found
            if results['n_seeds'] > 0:
                if output.suffix == '.h5' or output.suffix == '.hdf5':
                    extractor.save_hdf5(output)
                else:
                    extractor.save_csv(output, include_wavelengths=True)

                console.print(f"✓ Saved spectra to {output}")
            else:
                console.print("[yellow]⚠ No seeds found - no output file created[/yellow]")
                console.print("[yellow]  Try adjusting --min-pixels or --segmentation parameters[/yellow]")

            # Export mask if requested
            if export_mask:
                mask_path = output.parent / f"{output.stem}_mask.npy"
                segmenter.export_mask(mask_path, format="npy")
                console.print(f"✓ Saved mask to {mask_path}")

            # Export plots if requested
            if export_plots and results['n_seeds'] > 0:
                # Segmentation visualization with numbered seeds
                seg_plot = output.parent / f"{output.stem}_segmentation.png"
                fig = segmenter.visualize(
                    processed_data,
                    save_path=seg_plot,
                    show_labels=True,
                    show_boundaries=True
                )
                plt.close(fig)
                console.print(f"✓ Saved segmentation plot to {seg_plot}")

                # Seed distribution plot
                dist_plot = output.parent / f"{output.stem}_distribution.png"
                fig = extractor.plot_distribution(
                    save_path=dist_plot
                )
                plt.close(fig)
                console.print(f"✓ Saved distribution plot to {dist_plot}")

                # Spectra plot
                spec_plot = output.parent / f"{output.stem}_spectra.png"
                fig = extractor.plot_spectra(
                    save_path=spec_plot,
                    show_mean=True,
                    show_std=True
                )
                plt.close(fig)
                console.print(f"✓ Saved spectra plot to {spec_plot}")

            progress.update(task, completed=True)

        # Print summary
        console.print("\n[bold]Analysis Summary:[/bold]")
        if results['n_seeds'] > 0:
            stats = extractor.get_statistics()
        else:
            stats = {'n_bands': len(wavelengths) if wavelengths is not None else 0, 'global_mean': 0, 'global_std': 0, 'area_mean': 0}

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Seeds detected", str(n_seeds))
        table.add_row("Spectral bands", str(stats['n_bands']))
        table.add_row("Mean reflectance", f"{stats['global_mean']:.4f}")
        table.add_row("Std reflectance", f"{stats['global_std']:.4f}")
        table.add_row("Mean seed area", f"{stats['area_mean']:.0f} pixels")
        table.add_row("Output file", str(output))

        console.print(table)
        console.print("\n[bold green]✓ Analysis complete![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if ctx.obj['debug']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option(
    '-o', '--output',
    type=click.Path(path_type=Path),
    help='Output mask file path.'
)
@click.option(
    '--algorithm',
    type=click.Choice(['threshold', 'watershed', 'connected', 'combined']),
    default='watershed',
    help='Segmentation algorithm to use.'
)
@click.option(
    '--min-pixels',
    type=int,
    default=200,
    help='Minimum seed size in pixels.'
)
@click.option(
    '--visualize',
    is_flag=True,
    help='Show segmentation visualization.'
)
@click.pass_context
def segment(
    ctx,
    input_path: Path,
    output: Optional[Path],
    algorithm: str,
    min_pixels: int,
    visualize: bool
):
    """Perform seed segmentation only.

    Segments seeds from hyperspectral imagery without extracting spectra.

    Example:
        hyperseed segment dataset/sample --visualize
    """
    console.print(f"[bold green]Seed Segmentation[/bold green]")
    console.print(f"Input: {input_path}")

    try:
        # Load and calibrate data
        console.print("Loading and calibrating data...")
        calibrator = ReflectanceCalibrator(clip_negative=True, clip_max=1.0)
        calibrated_data, reader = calibrator.calibrate_from_directory(input_path)

        # Minimal preprocessing for segmentation
        console.print("Preprocessing...")
        settings = Settings()
        settings.preprocessing.method = "minimal"
        preprocessor = PreprocessingPipeline(settings.preprocessing)
        processed_data = preprocessor.fit_transform(calibrated_data)

        # Perform segmentation
        console.print(f"Segmenting with {algorithm} algorithm...")
        settings.segmentation.algorithm = algorithm
        settings.segmentation.min_pixels = min_pixels
        segmenter = SeedSegmenter(settings.segmentation)
        mask, n_seeds = segmenter.segment(processed_data, validate=True)

        console.print(f"✓ Found {n_seeds} seeds")

        # Save mask
        if output:
            segmenter.export_mask(output, format="npy")
            console.print(f"✓ Saved mask to {output}")

        # Visualize if requested
        if visualize:
            import matplotlib.pyplot as plt

            fig = segmenter.visualize(
                processed_data,
                show_labels=True,
                show_boundaries=True
            )
            plt.show()

        # Print statistics
        stats = segmenter.get_validation_stats()
        if stats:
            console.print(f"\nValidation Statistics:")
            console.print(f"  Initial seeds: {stats['initial_count']}")
            console.print(f"  Rejected (size): {stats['rejected_size']}")
            console.print(f"  Rejected (overlap): {stats['rejected_overlap']}")
            console.print(f"  Rejected (shape): {stats['rejected_shape']}")
            console.print(f"  Final seeds: {stats['final_count']}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        if ctx.obj['debug']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument('input_dir', type=click.Path(exists=True, path_type=Path))
@click.option(
    '-o', '--output-dir',
    type=click.Path(path_type=Path),
    help='Output directory for results.'
)
@click.option(
    '-c', '--config',
    type=click.Path(exists=True, path_type=Path),
    help='Configuration file path (YAML).'
)
@click.option(
    '--parallel',
    type=int,
    default=1,
    help='Number of parallel workers.'
)
@click.option(
    '--pattern',
    default='*',
    help='Pattern to match dataset directories.'
)
@click.option(
    '--min-pixels',
    type=int,
    default=200,
    help='Minimum seed size in pixels.'
)
@click.option(
    '--no-outlier-removal',
    is_flag=True,
    help='Disable automatic outlier removal.'
)
@click.pass_context
def batch(
    ctx,
    input_dir: Path,
    output_dir: Optional[Path],
    config: Optional[Path],
    parallel: int,
    pattern: str,
    min_pixels: int,
    no_outlier_removal: bool
):
    """Batch process multiple datasets.

    Processes all datasets in a directory matching the specified pattern.

    Example:
        hyperseed batch datasets/ --output-dir results/ --parallel 4
    """
    console.print(f"[bold green]Batch Processing[/bold green]")
    console.print(f"Input directory: {input_dir}")

    # Find all matching datasets
    datasets = list(input_dir.glob(pattern))
    datasets = [d for d in datasets if d.is_dir()]

    if not datasets:
        console.print(f"[bold red]No datasets found matching '{pattern}'[/bold red]")
        sys.exit(1)

    console.print(f"Found {len(datasets)} datasets to process")

    # Create output directory
    if output_dir is None:
        output_dir = input_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    if config:
        settings = Settings.load(config)
    else:
        settings = Settings()

    # Override min_pixels if different from default
    if min_pixels != 200:
        settings.segmentation.min_pixels = min_pixels

    # Process each dataset
    success_count = 0
    failed_datasets = []

    for idx, dataset in enumerate(datasets, 1):
        console.print(f"\n[{idx}/{len(datasets)}] Processing {dataset.name}...")

        try:
            # Create output path
            output_path = output_dir / f"{dataset.name}_spectra.csv"

            # Process dataset
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:

                # Calibration
                task = progress.add_task("Calibrating...", total=None)
                calibrator = ReflectanceCalibrator(
                    clip_negative=settings.calibration.clip_negative,
                    clip_max=settings.calibration.clip_max
                )
                calibrated_data, reader = calibrator.calibrate_from_directory(dataset)
                wavelengths = reader.get_wavelengths()
                progress.update(task, completed=True)

                # Preprocessing
                task = progress.add_task("Preprocessing...", total=None)
                preprocessor = PreprocessingPipeline(settings.preprocessing)
                processed_data = preprocessor.fit_transform(calibrated_data)
                progress.update(task, completed=True)

                # Segmentation
                task = progress.add_task("Segmenting...", total=None)
                segmenter = SeedSegmenter(settings.segmentation)
                mask, n_seeds = segmenter.segment(processed_data, validate=True)
                progress.update(task, completed=True)

                # Extraction
                task = progress.add_task("Extracting...", total=None)
                extractor = SpectralExtractor()
                results = extractor.extract(
                    calibrated_data,
                    mask,
                    wavelengths=wavelengths,
                    compute_stats=True
                )
                progress.update(task, completed=True)

                # Apply outlier removal if configured
                if settings.segmentation.remove_outliers and n_seeds > 0 and not no_outlier_removal:
                    outlier_config = {
                        'remove_outliers': settings.segmentation.remove_outliers,
                        'outlier_min_area': settings.segmentation.outlier_min_area,
                        'outlier_max_area': settings.segmentation.outlier_max_area,
                        'outlier_iqr_lower': settings.segmentation.outlier_iqr_lower,
                        'outlier_iqr_upper': settings.segmentation.outlier_iqr_upper,
                        'outlier_eccentricity': settings.segmentation.outlier_eccentricity,
                        'outlier_solidity': settings.segmentation.outlier_solidity,
                        'use_shape_filtering': settings.segmentation.use_shape_filtering
                    }

                    n_before = n_seeds
                    _, _, removed = extractor.remove_outliers(outlier_config, verbose=False)
                    n_seeds = len(extractor.seed_info)

                    if len(removed) > 0:
                        # Update results with filtered counts
                        results['n_seeds'] = n_seeds

                # Save results only if seeds were found
                if n_seeds > 0:
                    extractor.save_csv(output_path, include_wavelengths=True)

                    # Generate plots
                    plot_stem = output_path.stem.replace('_spectra', '')

                    # Seed distribution plot
                    dist_plot = output_path.parent / f"{plot_stem}_distribution.png"
                    fig = extractor.plot_distribution(save_path=dist_plot)
                    plt.close(fig)

                    # Spectral plot
                    spec_plot = output_path.parent / f"{plot_stem}_spectra.png"
                    fig = extractor.plot_spectra(
                        save_path=spec_plot,
                        show_mean=True,
                        show_std=True
                    )
                    plt.close(fig)

                    # Segmentation plot (optional, for detailed view)
                    seg_plot = output_path.parent / f"{plot_stem}_segmentation.png"
                    fig = segmenter.visualize(
                        processed_data,
                        save_path=seg_plot,
                        show_labels=True,
                        show_boundaries=True
                    )
                    plt.close(fig)

                    console.print(
                        f"  ✓ Processed: {n_seeds} seeds → {output_path.name}"
                    )
                    console.print(
                        f"     Generated visualizations:"
                    )
                    console.print(
                        f"       - {plot_stem}_distribution.png (spatial & size)"
                    )
                    console.print(
                        f"       - {plot_stem}_segmentation.png (numbered seeds)"
                    )
                    console.print(
                        f"       - {plot_stem}_spectra.png (spectral data)"
                    )
                else:
                    console.print(
                        f"  ⚠ No seeds found in {dataset.name} (check min-pixels threshold)"
                    )
            success_count += 1

        except Exception as e:
            console.print(f"  ✗ Failed: {e}")
            failed_datasets.append(dataset.name)
            if ctx.obj['debug']:
                import traceback
                traceback.print_exc()

    # Print summary
    console.print(f"\n[bold]Batch Processing Summary:[/bold]")
    console.print(f"  Successful: {success_count}/{len(datasets)}")
    if failed_datasets:
        console.print(f"  Failed: {', '.join(failed_datasets)}")

    if success_count == len(datasets):
        console.print("\n[bold green]✓ Batch processing complete![/bold green]")
    else:
        console.print("\n[bold yellow]⚠ Batch processing completed with errors[/bold yellow]")


@main.command()
@click.option(
    '-o', '--output',
    type=click.Path(path_type=Path),
    default='config.yaml',
    help='Output configuration file path.'
)
@click.option(
    '--preset',
    type=click.Choice(['minimal', 'standard', 'advanced']),
    default='standard',
    help='Configuration preset to use.'
)
def config(output: Path, preset: str):
    """Generate a configuration file.

    Creates a YAML configuration file with default or preset settings.

    Example:
        hyperseed config --output my_config.yaml --preset advanced
    """
    console.print(f"[bold green]Generating Configuration[/bold green]")

    settings = Settings()
    settings.apply_preset(preset)

    settings.save(output)
    console.print(f"✓ Saved configuration to {output}")
    console.print(f"  Preset: {preset}")
    console.print(f"  Preprocessing: {settings.preprocessing.get_step_names() if hasattr(settings.preprocessing, 'get_step_names') else preset}")
    console.print(f"  Segmentation: {settings.segmentation.algorithm}")
    console.print(f"  Min seed size: {settings.segmentation.min_pixels} pixels")


@main.command()
def info():
    """Display information about hyperseed.

    Shows version, dependencies, and system information.
    """
    console.print(f"[bold green]Hyperseed Information[/bold green]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Component", style="cyan")
    table.add_column("Version/Info", justify="right")

    table.add_row("Hyperseed version", __version__)

    # Python version
    import sys
    table.add_row("Python version", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

    # Key dependencies
    try:
        import numpy
        table.add_row("NumPy", numpy.__version__)
    except:
        pass

    try:
        import scipy
        table.add_row("SciPy", scipy.__version__)
    except:
        pass

    try:
        import skimage
        table.add_row("scikit-image", skimage.__version__)
    except:
        pass

    try:
        import spectral
        table.add_row("Spectral", spectral.__version__)
    except:
        pass

    # System info
    import platform
    table.add_row("Platform", platform.platform())
    table.add_row("Processor", platform.processor() or "N/A")

    # GPU support
    gpu_status = "Not available"
    try:
        import torch
        if torch.backends.mps.is_available():
            gpu_status = "Metal Performance Shaders"
        elif torch.cuda.is_available():
            gpu_status = f"CUDA ({torch.cuda.device_count()} devices)"
    except:
        pass
    table.add_row("GPU acceleration", gpu_status)

    console.print(table)

    console.print("\n[bold]Documentation:[/bold]")
    console.print("  GitHub: https://github.com/yourusername/hyperseed")
    console.print("  Docs: https://hyperseed.readthedocs.io")


if __name__ == "__main__":
    main()