use ndarray::ErrorKind::IncompatibleShape;
use ndarray::{
    s, Array, Array1, Array2, Array3, Array4, ArrayView2, ArrayView3, ArrayView4, Axis, Dimension,
    NewAxis, ShapeBuilder, ShapeError, Zip,
};
use ndarray_linalg::{Eigh, Norm, UPLO};
use ndarray_rand::rand::{thread_rng, Rng};
use ndarray_rand::rand_distr::{Distribution, Normal, StandardNormal};
use ndrustfft::{ndfft_par, ndfft_r2c_par, ndifft_par, FftHandler, R2cFftHandler, Zero};
use num_complex::{Complex, Complex32, ComplexDistribution};
use num_traits::{Float, FloatConst, Inv};
use std::ops::{Mul, Range};

extern crate blas_src;

fn cos_window(n: usize, c0: f32, c1: f32) -> Array1<f32> {
    let step = 2. * f32::PI() / (n - 1) as f32;
    c0 - c1 * Array::range(0., 2. * f32::PI() + step, step).mapv(f32::cos)
}

pub fn hann(n: usize) -> Array1<f32> {
    cos_window(n, 0.5, 0.5)
}

// Samples in second dimension
pub fn fft_windowed(
    samples: ArrayView2<f32>,
    window_size: usize,
    step_size: usize,
) -> Array3<Complex<f32>> {
    let time_frames: Vec<ArrayView2<f32>> = samples
        .axis_windows(Axis(1), window_size)
        .into_iter()
        .step_by(step_size)
        .collect(); // TODO: Do this lazily.
    let channels = samples.shape()[0];
    let num_frq_bin = window_size / 2 + 1;

    let mut fft_array: Array3<Complex<f32>> =
        Array::zeros((time_frames.len(), channels, num_frq_bin));
    let mut fft_handler = R2cFftHandler::<f32>::new(window_size);

    let hann_win = hann(window_size);

    Zip::from(fft_array.axis_iter_mut(Axis(0)))
        .and(&time_frames)
        .for_each(|mut out, data| {
            ndfft_r2c_par(&(data * &hann_win), &mut out, &mut fft_handler, 1);
        });
    fft_array
}

#[allow(dead_code)]
pub fn random_real<D, Sh, F>(shape: Sh, mean: F, std: F) -> Array<F, D>
where
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
    F: Float + Send + Sync,
    StandardNormal: Distribution<F>,
{
    let distr = Normal::new(mean, std).unwrap();
    let mut matrix: Array<F, D> = Array::zeros(shape);
    matrix.par_mapv_inplace(|_| thread_rng().sample(distr));
    matrix
}
#[allow(dead_code)]
pub fn random_complex<D, Sh, F>(shape: Sh, mean: F, std: F) -> Array<Complex<F>, D>
where
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
    F: Float + Send + Sync,
    StandardNormal: Distribution<F>,
{
    let re = Normal::new(mean, std).unwrap();
    let im = Normal::new(mean, std).unwrap();
    let distr = ComplexDistribution::new(re, im);
    let mut matrix: Array<Complex<F>, D> = Array::zeros(shape);
    matrix.par_mapv_inplace(|_| thread_rng().sample(distr));
    matrix
}

#[allow(dead_code)]
pub fn wideband_music(
    data: ArrayView2<f32>,
    time_win: usize,
    steering: ArrayView3<Complex32>,
    frq_bins: Range<usize>,
    num_sources: usize,
    channel_step: Option<usize>,
    time_step: Option<usize>,
) -> Result<Array4<f32>, ShapeError> {
    let num_frq_bins = steering.shape()[0];
    let num_look_directions = steering.shape()[2];
    let channel_win = steering.shape()[1];
    let channel_step = channel_step.unwrap_or(channel_win);
    let time_step = time_step.unwrap_or(time_win);

    if frq_bins.len() != num_frq_bins {
        return Err(ShapeError::from_kind(IncompatibleShape));
    }
    if frq_bins.end > time_win / 2 {
        println!(
            "Max frq bin index {} bigger than biggest frq bin {}",
            frq_bins.end,
            time_win / 2
        );
        return Err(ShapeError::from_kind(IncompatibleShape));
    }

    //println!("channel step:{:?}, time step:{:?}", channel_step, time_step);

    // FFT tensor:  time_frames.len() x channels x num_frq_bin
    let binding = fft_windowed(data, time_win, time_step);
    let mut fft: ArrayView4<Complex32> = binding.slice(s![.., .., frq_bins, NewAxis]);
    fft.swap_axes(1, 2);

    let num_time_frames = fft.shape()[0];

    // Select frequency bins, swap time and frequency axis, and add dimesion.
    // This is more efficient than doing it inside loops.

    let sub_array_vec: Vec<ArrayView4<Complex32>> = fft
        .axis_windows(Axis(2), channel_win)
        .into_iter()
        .step_by(channel_step)
        .collect();
    let num_sub_arrays = sub_array_vec.len();

    let mut doa: Array4<f32> = Array::zeros((
        num_sub_arrays,
        num_time_frames,
        num_frq_bins,
        num_look_directions,
    ));

    Zip::from(doa.axis_iter_mut(Axis(0)))
        .and(&sub_array_vec)
        .par_for_each(|mut doa_subarray, x_subarray| {
            for (mut doa_time, x_time) in doa_subarray
                .axis_iter_mut(Axis(0))
                .zip(x_subarray.axis_iter(Axis(0)))
            {
                Zip::from(doa_time.axis_iter_mut(Axis(0)))
                    .and(x_time.axis_iter(Axis(0)))
                    .and(steering.axis_iter(Axis(0)))
                    .for_each(|mut look, x, steer| {
                        // Calculate sample covariance matrix and noise subspace
                        let cov = x.dot(&x.mapv(|v| v.conj()).t());
                        let (_, vec) = cov.eigh(UPLO::Lower).unwrap();
                        let noise_ss = vec.slice(s![.., ..channel_win - num_sources]);
                        let norm = noise_ss
                            .mapv(|v| v.conj())
                            .t()
                            .dot(&steer)
                            // Powi removed for performace reasons
                            .map_axis(Axis(0), |view| view.norm_l2().powi(2).inv());
                        look.assign(&norm);
                    });
            }
        });
    Ok(doa)
}

pub fn hilbert(
    data: ArrayView2<f32>,
    frequency_range: Option<(f32, f32)>,
    fs: Option<f32>,
) -> Array2<Complex32> {
    let input_shape = data.raw_dim();
    let n: usize = input_shape[1];
    let fs = fs.unwrap_or(1f32);

    let mut fft_handler = FftHandler::<f32>::new(n);
    let mut input: Array2<Complex32> = data.map(|&x| Complex32::new(x, 0.0));
    let mut output: Array2<Complex32> = Array::zeros(input_shape);

    ndfft_par(&input, &mut output, &mut fft_handler, 1);

    if let Some((from, to)) = frequency_range {
        let mut from = (n as f32 * from / fs) as usize;
        let to = (n as f32 * to / fs) as usize;
        assert!(from < to);
        output
            .slice_mut(s![.., ..from])
            .par_map_inplace(|x| x.set_zero());
        output
            .slice_mut(s![.., to..])
            .par_map_inplace(|x| x.set_zero());

        if from == 0 {
            from = 1;
        }
        output
            .slice_mut(s![.., from..to])
            .par_mapv_inplace(|x| x.mul(2.0));
    } else {
        output
            .slice_mut(s![.., n / 2..])
            .par_map_inplace(|x| x.set_zero());
        output
            .slice_mut(s![.., 1..n / 2])
            .par_mapv_inplace(|x| x.mul(2.0));
    }

    ndifft_par(&output, &mut input, &mut fft_handler, 1);

    input
}

#[allow(dead_code)]
pub fn music(
    data: ArrayView2<Complex32>,
    time_win: usize,
    steering: ArrayView2<Complex32>,
    num_sources: usize,
    channel_step: Option<usize>,
    time_step: Option<usize>,
) -> Result<Array3<f32>, ShapeError> {
    let num_look_directions = steering.shape()[1];
    let channel_win = steering.shape()[0];
    let channel_step = channel_step.unwrap_or(channel_win);
    let time_step = time_step.unwrap_or(time_win);

    // Form sample covariance matrices.
    let virtual_arrays: Vec<ArrayView2<Complex32>> = data
        .axis_windows(Axis(0), channel_win)
        .into_iter()
        .step_by(channel_step)
        .collect();
    let num_sub_arrays = virtual_arrays.len();
    let num_time_frames = (data.shape()[1] - time_win) / time_step + 1;

    let mut doa: Array3<f32> = Array::zeros((num_sub_arrays, num_time_frames, num_look_directions));

    // for each array
    Zip::from(doa.axis_iter_mut(Axis(0)))
        .and(&virtual_arrays)
        .par_for_each(|mut doa_subarray, x_subarray| {
            for (x, mut doa_frame) in x_subarray
                .axis_windows(Axis(1), time_win)
                .into_iter()
                .step_by(time_step)
                .zip(doa_subarray.axis_iter_mut(Axis(0)))
            {
                let cov = x.dot(&x.mapv(|v| v.conj()).t());
                let (_, vec) = cov.eigh(UPLO::Lower).unwrap();
                let noise_ss = vec.slice(s![.., ..channel_win - num_sources]);
                let norm = noise_ss
                    .mapv(|v| v.conj())
                    .t()
                    .dot(&steering)
                    .map_axis(Axis(0), |view| view.norm_l2().powi(2).inv());
                doa_frame.assign(&norm);
            }
        });
    Ok(doa)
}
