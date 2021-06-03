"""This code has been took from https://github.com/danielnewman09/Kurtogram-Analysis/"

Implements the kurtogram routine of Antoni (2005).
"""

import logging

import numpy as np

from scipy.signal import firwin
from scipy.signal import lfilter

logger = logging.getLogger(__name__)


eps = np.finfo(float).eps

VARIANCE_ENVELOPE_MAGNITUDE = "kurt1"
KURTOSIS_COMPLEX_ENVELOPE = "kurt2"


def get_h_parameters(NFIR: int, fcut: float):
    """Calculates h-parameters used in Antoni (2005)

    Parameters
    ----------
    NFIR: int
        Length of FIR filter
    fcut: float
        fraction of Nyquist for filter

    Returns
    -------
    np.array
        h-parameters: h, g, h1, h2, h3
    """

    h = firwin(NFIR + 1, fcut) * np.exp(2 * 1j * np.pi * np.arange(NFIR + 1) * 0.125)
    n = np.arange(2, NFIR + 2)
    g = np.power(h[(1 - n) % NFIR] * (-1), (1 - n))
    NFIR = int(np.fix((3.0 / 2.0 * NFIR)))
    h1 = firwin(NFIR + 1, 2.0 / 3 * fcut) * np.exp(
        2j * np.pi * np.arange(NFIR + 1) * 0.25 / 3.0
    )
    h2 = h1 * np.exp(2j * np.pi * np.arange(NFIR + 1) / 6.0)
    h3 = h1 * np.exp(2j * np.pi * np.arange(NFIR + 1) / 3.0)
    return (h, g, h1, h2, h3)


def getBandwidthAndFrequency(
    nlevel: int, Fs: float, level_w: np.array, level_index: int, freq_index: int
):
    """Compute the bandwidth and frequency parameters

    Parameters
    -----------
    nlevel: int
        number of decomposition levels
    Fs: float
        sampling frequency of the signal
    level_w: np.arary
        vector of decomposition levels
    freq_w: np.array
        vector of frequencies
    level_index: int
        index of the level
    freq_index: int
        index of the frequency

    Returns
    -------
    bw, fc, fi, l1
        * bw: bandwidth
        * fc: central frequency
        * fi: index of the frequency sequence within the level l1
        * l1: level
    """

    l1 = level_w[level_index]
    fi = (freq_index) / 3.0 / 2 ** (nlevel + 1)
    fi += 2.0 ** (-2 - l1)
    bw = Fs * 2 ** -(l1) / 2
    fc = Fs * fi

    return bw, fc, fi, l1


def get_GridMax(grid):
    """Gets maximum of a nD grid and its unraveled index

    Parameters
    ----------
    grid: np.array
        an nD-grid

    Returns
    --------
    * M : grid maximum
    * index : index of maximum in unraveled grid
    """

    index = np.argmax(grid)
    M = np.amax(grid)
    index = np.unravel_index(index, grid.shape)

    return M, index


def fast_kurtogram(
    x: np.array,
    fs:float,
    nlevel: int
):
    """Computes the fast kurtogram  of a signal  up to a level

    Maximum number of decomposition levels is log2(length(x)), but it is
    recommended to stay by a factor 1/8 below this.

    Also returns the vector of k-levels Level_w, the frequency vector
    freq_w, the complex envelope of the signal c and the extreme
    frequencies of the "best" bandpass f_lower and f_upper.
    J. Antoni : 02/2005


    Parameters
    ----------
    x: np.array
        Signal to analyse
    nlevel: int
        number of decomposition levels
    Fs: integer
        Sampling frequency of signal x
    NFIR: integer
        Length of FIR filter
    fcut: float
        Fraction of Nyquist for filter
    option: str
        Possible values are the following constants:
        * VARIANCE_ENVELOPE_MAGNITUDE: classical kurtosis based on 4th order statistics
        * KURTOSIS_COMPLEX_ENVELOPE: robust kurtosis based on 2nd order statistics of the
            envelope (if there is any difference in the kurtogram between the
            two measures, this is due to the presence of impulsive additive
            noise)


    Returns
    --------
    Kwav, Level_w, freq_w, c, f_lower, f_upper
        * Kwav: kurtogram
        * Level_w: vector of levels
        * freq_w: frequency vector
        * c: complex envelope of the signal filtered in the frequency band that maximizes the kurtogram
        * f_lower: lower frequency of the band pass
        * f_upper: upper frequency of the band pass
    """

    N = x.flatten().size
    N2 = np.log2(N) - 7
    if nlevel > N2:
        logger.error("Please enter a smaller number of decomposition levels")

    x = x- np.mean(x)
    N = 16
    fc = 0.4

    N = 16
    fc = 0.4
    
    h = firwin(N+1,fc) * np.exp(2j * np.pi * np.arange(N+1) * 0.125)
    
    n = np.arange(2,N+2)
    
    g = h[(1-n) % N] * (-1.)**(1-n)
    
    N = int(np.fix(3/2*N))
    
    h1 = firwin(N+1,2/3 * fc) * np.exp(2j * np.pi * np.arange(0,N+1) * 0.25/3)
    h2 = h1 * np.exp(2j * np.pi * np.arange(0,N+1) / 6)
    h3 = h1 * np.exp(2j * np.pi * np.arange(0,N+1) / 3)

    Kwav = K_wpQ(x,h,g,h1,h2,h3,nlevel,'kurt2')
    Kwav = np.clip(Kwav,0,np.inf)
    Level_w = np.arange(1,nlevel+1)
    Level_w = np.vstack((Level_w,
                         Level_w + np.log2(3)-1)).flatten()
    Level_w = np.sort(np.insert(Level_w,0,0)[:2*nlevel])

    freq_w = fs*(np.arange(3*2**nlevel)/(3*2**(nlevel+1)) + 1/(3*2**(2+nlevel)))
    
    max_level_index = np.argmax(Kwav[np.arange(Kwav.shape[0]),np.argmax(Kwav,axis=1)])
    max_kurt = np.amax(Kwav[np.arange(Kwav.shape[0]),np.argmax(Kwav,axis=1)])

    level_max = Level_w[max_level_index]

    bandwidth = fs*2**(-(Level_w[max_level_index] + 1))

    J = np.argmax(Kwav[max_level_index,:])
    fc = freq_w[J]

    return Kwav, freq_w, Level_w, bandwidth, fc


def K_wpQ(
    x: np.array,
    h: np.array,
    g: np.array,
    h1: np.array,
    h2: np.array,
    h3: np.array,
    nlevel: int,
    opt: str,
    level: int = 0,
):
    """Calculates the kurtosis K (2-D matrix) of the complete quinte wavelet packet
    transform w of signal x, up to nlevel, using the lowpass and highpass filters
    h and g, respectively. The WP coefficients are sorted according to the frequency
    decomposition. This version handles both real and analytical filters, but
    does not yield WP coefficients suitable for signal synthesis.
    J. Antoni : 12/2004
    Translation to Python: T. Lecocq 02/2012

    Parameters
    ----------
    x: np.array
        signal
    h: np.array
        lowpass filter
    g: np.array
        higpass filter
    h1: np.array
        filter parameter returned by get_h_parameters
    h2: np.array
        filter parameter returned by get_h_parameters
    h3: np.array
        filter parameter returned by get_h_parameters
    nlevel: int
        number of decomposition levels
    opt: str
        Possible values are:
        * 'kurt1' = variance of the envelope magnitude
        * 'kurt2' = kurtosis of the complex envelope
    level: int
        Current decomposition level for this call

    Returns
    --------
    kurtosis
    """

    L = np.floor(np.log2(len(x)))
    if level == 0:
        if nlevel >= L:
            logging.error("nlevel must be smaller")
        level = nlevel
    x = x.ravel()
    KD, KQ = K_wpQ_local(x, h, g, h1, h2, h3, nlevel, opt, level)
    K = np.zeros((2 * nlevel, 3 * 2 ** nlevel))

    K[0, :] = KD[0, :]
    for i in range(1, nlevel):
        K[2 * i - 1, :] = KD[i, :]
        K[2 * i, :] = KQ[i - 1, :]

    K[2 * nlevel - 1, :] = KD[nlevel, :]
    return K


def K_wpQ_local(
    x: np.array,
    h: np.array,
    g: np.array,
    h1: np.array,
    h2: np.array,
    h3: np.array,
    nlevel: int,
    opt: str,
    level: int,
):
    """Computes the 2-D vector which contains the kurtosis value of the signal.

    Also compute  the 2 kurtosis values corresponding to the signal filtered into 2 different
    band-passes and computes the 2-D vector KQ which contains the 3 kurtosis values corresponding
    to the signal filtered into 3 different band-passes.

    Is a recursive function.

    Parameters
    -----------
    x: np.array
        signal
    h: np.array
        lowpass filter
    g: np.array
        highpass filter
    h1: np.array
        filter parameter returned by get_h_parameters
    h2: np.array
        filter parameter returned by get_h_parameters
    h3: np.array
        filter parameter returned by get_h_parameters
    nlevel: int
        number of decomposition levels
    opt: str.
        Possible values are: 'kurt1' or 'kurt2']
        * 'kurt1' = variance of the envelope magnitude
        * 'kurt2' = kurtosis of the complex envelope
    level: int
        Current decomposition level

    Return
    -------
    K
    Kq
    """

    a, d = DBFB(x, h, g)

    N = len(a)
    d = d * np.power(-1.0, np.arange(1, N + 1))  # indices pairs multipliés par -1
    K1 = kurt(a[len(h) - 1 :], opt)
    K2 = kurt(d[len(g) - 1 :], opt)

    if level > 2:
        a1, a2, a3 = TBFB(a, h1, h2, h3)
        d1, d2, d3 = TBFB(d, h1, h2, h3)
        Ka1 = kurt(a1[len(h) - 1 :], opt)
        Ka2 = kurt(a2[len(h) - 1 :], opt)
        Ka3 = kurt(a3[len(h) - 1 :], opt)
        Kd1 = kurt(d1[len(h) - 1 :], opt)
        Kd2 = kurt(d2[len(h) - 1 :], opt)
        Kd3 = kurt(d3[len(h) - 1 :], opt)
    else:
        Ka1 = 0
        Ka2 = 0
        Ka3 = 0
        Kd1 = 0
        Kd2 = 0
        Kd3 = 0

    if level == 1:
        K = np.array([K1 * np.ones(3), K2 * np.ones(3)]).flatten()
        KQ = np.array([Ka1, Ka2, Ka3, Kd1, Kd2, Kd3])
    if level > 1:
        Ka, KaQ = K_wpQ_local(a, h, g, h1, h2, h3, nlevel, opt, level - 1)

        Kd, KdQ = K_wpQ_local(d, h, g, h1, h2, h3, nlevel, opt, level - 1)

        K1 = K1 * np.ones(np.max(Ka.shape))
        K2 = K2 * np.ones(np.max(Kd.shape))
        K12 = np.append(K1, K2)
        Kad = np.hstack((Ka, Kd))
        K = np.vstack((K12, Kad))

        Long = int(2.0 / 6 * np.max(KaQ.shape))
        Ka1 = Ka1 * np.ones(Long)
        Ka2 = Ka2 * np.ones(Long)
        Ka3 = Ka3 * np.ones(Long)
        Kd1 = Kd1 * np.ones(Long)
        Kd2 = Kd2 * np.ones(Long)
        Kd3 = Kd3 * np.ones(Long)
        tmp = np.hstack((KaQ, KdQ))

        KQ = np.concatenate((Ka1, Ka2, Ka3, Kd1, Kd2, Kd3))
        KQ = np.vstack((KQ, tmp))

    if level == nlevel:
        K1 = kurt(x, opt)
        K = np.vstack((K1 * np.ones(np.max(K.shape)), K))

        a1, a2, a3 = TBFB(x, h1, h2, h3)
        Ka1 = kurt(a1[len(h) - 1 :], opt)
        Ka2 = kurt(a2[len(h) - 1 :], opt)
        Ka3 = kurt(a3[len(h) - 1 :], opt)
        Long = int(1.0 / 3 * np.max(KQ.shape))
        Ka1 = Ka1 * np.ones(Long)
        Ka2 = Ka2 * np.ones(Long)
        Ka3 = Ka3 * np.ones(Long)
        tmp = np.array(KQ[0:-2])

        KQ = np.concatenate((Ka1, Ka2, Ka3))
        KQ = np.vstack((KQ, tmp))

    return K, KQ


def kurt(x: np.array, opt: str):
    """Calculates kurtosis of a signal according to the option chosen

    Parameters
    ----------
    x: np.array
        Input signal
    opt: str
    Possible values are the following constants

        * 'VARIANCE_ENVELOPE_MAGNITUDE = variance of the envelope magnitude
        * 'KURTOSIS_COMPLEX_ENVELOPE = kurtosis of the complex envelope


    Returns
    -------
    float:
        Kurtosis
    """
    if opt == "kurt2":
        if np.all(x == 0):
            K = 0
            E = 0
            return K
        x = x - np.mean(x)
        E = np.mean(np.abs(x) ** 2)
        if E < eps:
            K = 0
            return K

        K = np.mean(np.abs(x) ** 4) / E ** 2

        if np.all(np.isreal(x)):
            K = K - 3
        else:
            K = K - 2

    if opt == "kurt1":
        if np.all(x == 0):
            K = 0
            E = 0
            return K
        x = x - np.mean(x)
        E = np.mean(np.abs(x))
        if E < eps:
            K = 0
            return K

        K = np.mean(np.abs(x) ** 2) / E ** 2

        if np.all(np.isreal(x)):
            K = K - 1.57
        else:
            K = K - 1.27

    return K


def DBFB(x: np.array, h: np.array, g: np.array):
    """Double-band filter-bank.

    coefficients vector a and detail coefficients vector d,
    obtained by passing signal x though a two-band analysis filter-bank.

    Parameters
    ----------
    x: np.array
        Input signal

    h: np.array
        The decomposition low-pass filter
    g: np.array
        The decomposition high-pass filter.

    Returns
    -------
    Tuple[np.array, np.array]

    """

    # lowpass filter
    try:
        a = lfilter(h, 1, x)
        a = a[1::2]
        a = a.ravel()
    except:        
        a = 0

    # highpass filter
    d = lfilter(g, 1, x)
    d = d[1::2]
    d = d.ravel()

    return (a, d)


def TBFB(x, h1, h2, h3):
    """Triple-band filter-bank.

    Parameters
    -----------
    x: np.array
    h1: np.array
        filter parameter
    h2: np.array
        filter parameter
    h3: np.array
        filter parameter

    Returns
    --------
    Tuple[np.array, np.array, np.array]
    """

    # lowpass filter
    a1 = lfilter(h1, 1, x)
    a1 = a1[2::3]
    a1 = a1.ravel()

    # passband filter
    a2 = lfilter(h2, 1, x)
    a2 = a2[2::3]
    a2 = a2.ravel()

    # highpass filter
    a3 = lfilter(h3, 1, x)
    a3 = a3[2::3]
    a3 = a3.ravel()

    return (a1, a2, a3)


def Find_wav_kurt(x:np.array, h:np.array, g:np.array, h1:np.array, h2:np.array, h3:np.array,  Sc:float, Fr:float, Fs:int=1):
    """:returns: c, s, threshold, Bw, fc

    Parameters
    ----------
    x : np.array
    h : np.array
    g : np.array
    h1 : np.array
    h2 : np.array
    h3 : np.array
    Sc : float
    Fr : float
    Fs : int, optional
        by default 1

    Returns
    -------
    Tuple
       c, s, threshold, Bw, fc
    """
    level = np.fix((Sc)) + ((Sc % 1) >= 0.5) * (np.log2(3) - 1)
    Bw = 2 ** (-level - 1)
    freq_w = np.arange(0, 2 ** level) / 2 ** (level + 1) + Bw / 2.0
    J = np.argmin(np.abs(freq_w - Fr))
    fc = freq_w[J]
    i = int(np.round(fc / Bw - 1.0 / 2))
    if level % 1 == 0:
        acoeff = binary(i, int(level))
        bcoeff = []
        temp_level = level
    else:
        i2 = int(np.fix((i / 3.0)))
        temp_level = np.fix((level)) - 1
        acoeff = binary(i2, int(temp_level))
        bcoeff = i - i2 * 3
    acoeff = acoeff[::-1]
    c = K_wpQ_filt(x, h, g, h1, h2, h3, acoeff, bcoeff, temp_level)

    t = np.arange(len(x)) / float(Fs)
    tc = np.linspace(t[0], t[-1], len(c))
    s = np.real(c * np.exp(2j * np.pi * fc * Fs * tc))

    sig = np.median(np.abs(c)) / np.sqrt(np.pi / 2.0)
    threshold = sig * raylinv(
        np.array(
            [
                0.999,
            ]
        ),
        np.array(
            [
                1,
            ]
        ),
    )

    return c, s, threshold, Bw, fc


def binary(i: int, k: int):
    """Computes the coefficients of the binary expansion of i

    i = a(1)*2^(k-1) + a(2)*2^(k-2) + ... + a(k)

    Parameters
    ----------
    i: int
        integer to expand
    k: int
        nummber of coefficients

    Returns
    --------
    np.array:
        coefficients
    """

    if i >= 2 ** k:
        logging.error("i must be such that i < 2^k !!")

    a = np.zeros(k)
    temp = i
    for l in np.arange(k - 1, -1, -1):
        a[k - l - 1] = int(np.fix(temp / 2 ** l))
        temp = temp - int(np.fix(a[k - l - 1] * 2 ** l))

    return a


def K_wpQ_filt(x, h, g, h1, h2, h3, acoeff, bcoeff, level=0):
    """
    Calculates the kurtosis K of the complete quinte wavelet packet transform w
    of signal x, up to nlevel, using the lowpass and highpass filters h and g,
    respectively. The WP coefficients are sorted according to the frequency
    decomposition.
    This version handles both real and analytical filters, but does not yield
    WP coefficients suitable for signal synthesis.
    J. Antoni : 12/2004
    Translation to Python: T. Lecocq 02/2012

    Parameters
    ----------
    x: np.array
    h: np.array
    g: np.array
    h1: np.array
    h2: np.array
    h3: np.array
    acoeff
    bcoeff
    level: int

    """

    nlevel = len(acoeff)
    L = np.floor(np.log2(len(x)))
    if level == 0:
        if nlevel >= L:
            logging.error("nlevel must be smaller !!")
        level = nlevel
    x = x.ravel()
    if nlevel == 0:
        if bcoeff == []:
            c = x
        else:
            c1, c2, c3 = TBFB(x, h1, h2, h3)
            if bcoeff == 0:
                c = c1[len(h1) - 1 :]
            elif bcoeff == 1:
                c = c2[len(h2) - 1 :]
            elif bcoeff == 2:
                c = c3[len(h3) - 1 :]
    else:
        c = _K_wpQ_filt_local(x, h, g, h1, h2, h3, acoeff, bcoeff, level)
    return c


def _K_wpQ_filt_local(x, h, g, h1, h2, h3, acoeff, bcoeff, level):
    """Performs one analysis level into the analysis tree"""

    a, d = DBFB(x, h, g)
    N = len(a)
    d = d * np.power(-1.0, np.arange(1, N + 1))
    level = int(level)
    if level == 1:
        if bcoeff == []:
            if acoeff[level - 1] == 0:
                c = a[len(h) - 1 :]
            else:
                c = d[len(g) - 1 :]
        else:
            if acoeff[level - 1] == 0:
                c1, c2, c3 = TBFB(a, h1, h2, h3)
            else:
                c1, c2, c3 = TBFB(d, h1, h2, h3)
            if bcoeff == 0:
                c = c1[len(h1) - 1 :]
            elif bcoeff == 1:
                c = c2[len(h2) - 1 :]
            elif bcoeff == 2:
                c = c3[len(h3) - 1 :]
    if level > 1:
        if acoeff[level - 1] == 0:
            c = _K_wpQ_filt_local(a, h, g, h1, h2, h3, acoeff, bcoeff, level - 1)
        else:
            c = _K_wpQ_filt_local(d, h, g, h1, h2, h3, acoeff, bcoeff, level - 1)

    return c


def raylinv(p, b):
    """Inverse of the Rayleigh cumulative distribution function (cdf).

    X = RAYLINV(P,B) returns the Rayleigh cumulative distribution function
    with parameter B at the probabilities in P.

    Parameters
    ---------
    p: np.array
        Vector of probabilities
    b: float
        Parameter of the distribution

    Returns
    -------
    float
    """

    x = np.zeros(len(p))

    k = np.where(((b <= 0) | (p < 0) | (p > 1)))[0]

    if len(k) != 0:
        tmp = np.NaN
        x[k] = tmp(len(k))

    k = np.where(p == 1)[0]
    if len(k) != 0:
        tmp = np.Inf
        x[k] = tmp(len(k))

    k = np.where(((b > 0) & (p > 0) & (p < 1)))[0]

    if len(k) != 0:
        pk = p[k]
        bk = b[k]
        x[k] = np.sqrt((-2 * bk ** 2) * np.log(1 - pk))

    return x
