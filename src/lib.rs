use numpy::{
    Complex32, IntoPyArray, PyArray2, PyArray3, PyArray4, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::prelude::*;

mod music;

///MUSIC pseudo spectrum
///Splits data into virtual arrays and time frames.
///
///Parameters
///----------
///data : numpy.ndarray   float32
///    data matrix of shape '(n, m)' where 'n' is the spatial, and 'm' is the temporal dimension.
///steering: numpy.ndarray complex64
///    steering tensor. First dimesion is frequency bins, second is channels per virtual array and third is look directions
///time_win: int
///    same as fft size
///frq_bin_start: int
///    Range start of frequency bins to keep
///frq_bin_stop: int
///    Range stop of frequency bins to keep (non inclusive)
///channel_step: int, default= virtual array size. (no overlap)
///    steps for sliding window in channels. default the same as channel_win, no overlap.
///time_step: int, default=time_win (no overlap)
///    steps for sliding window in time. default the same as time_win, no overlap.
///Returns
///-------
///ndarray float32
///    virtual arrays x time frames x frequency bins x look directions.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn wideband_music<'a>(
    py: Python<'a>,
    data: PyReadonlyArray2<'a, f32>,
    steering: PyReadonlyArray3<'a, Complex32>,
    time_win: usize,
    frq_bin_start: usize,
    frq_bin_stop: usize,
    num_sources: usize,
    channel_step: Option<usize>,
    time_step: Option<usize>,
) -> Bound<'a, PyArray4<f32>> {
    let doa = music::wideband_music(
        data.as_array(),
        time_win,
        steering.as_array(),
        frq_bin_start..frq_bin_stop,
        num_sources,
        channel_step,
        time_step,
    )
    .unwrap();
    doa.into_pyarray_bound(py)
}

///MUSIC pseudo spectrum
///Splits data into virtual arrays and time frames.
///
///Parameters
///----------
///data : numpy.ndarray   complex64
///    data matrix of shape '(n, m)' where 'n' is the spatial, and 'm' is the temporal dimension.
///steering: numpy.ndarray complex64
///    steering tensor. First dimesion is frequency bins, second is channels per virtual array and third is look directions
///time_win: int
///    same as fft size
///    Range stop of frequency bins to keep (non inclusive)
///channel_step: int, default= virtual array size. (no overlap)
///    steps for sliding window in channels. default the same as channel_win, no overlap.
///time_step: int, default=time_win (no overlap)
///    steps for sliding window in time. default the same as time_win, no overlap.
///Returns
///-------
///ndarray float32
///    virtual arrays x time frames x look directions.
#[pyfunction]
#[pyo3(name = "music")]
fn _music<'a>(
    py: Python<'a>,
    data: PyReadonlyArray2<'a, Complex32>,
    steering: PyReadonlyArray2<'a, Complex32>,
    time_win: usize,
    num_sources: usize,
    channel_step: Option<usize>,
    time_step: Option<usize>,
) -> Bound<'a, PyArray3<f32>> {
    let doa = music::music(
        data.as_array(),
        time_win,
        steering.as_array(),
        num_sources,
        channel_step,
        time_step,
    )
    .unwrap();
    doa.into_pyarray_bound(py)
}

/// Hilbert transform
///Calculates the analytical signal from real signal.
///
///
///Parameters
///----------
///data : numpy.ndarray float32
///    data matrix of shape '(n, m)' where 'n' is the spatial, and 'm' is the temporal dimension.
///    Enshuring 'm' is a power may increase performance.
///frequency_range : (float, float) = None
///    Start and stop frequency.
///    set samplerate if not normalized frequencies are desired eg. in range [0, 0.5>
///sample_rate : float = None
///    Temporal samplerate in Hz. Defaults to 1.
///Returns
///-------
///result : numpy.ndarray Complex64
///    Analytical signal
///
#[pyfunction]
fn hilbert<'a>(
    py: Python<'a>,
    data: PyReadonlyArray2<'a, f32>,
    frequency_range: Option<(f32, f32)>,
    sample_rate: Option<f32>,
) -> Bound<'a, PyArray2<Complex32>> {
    music::hilbert(data.as_array(), frequency_range, sample_rate).into_pyarray_bound(py)
}

/// A Python module implemented in Rust.
#[pymodule]
fn dasrust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    //m.add_function(wrap_pyfunction!(test64, m)?)?;
    m.add_function(wrap_pyfunction!(wideband_music, m)?)?;
    m.add_function(wrap_pyfunction!(hilbert, m)?)?;
    m.add_function(wrap_pyfunction!(_music, m)?)?;
    Ok(())
}
