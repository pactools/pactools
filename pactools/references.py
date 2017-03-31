"""This is the list of the bibliographic references of the methods implemented
in this package.

Examples
--------
>>> from pactools import REFERENCES
>>> print(REFERENCES['duprelatour'])
[Dupre la Tour & al 2017]

"""


class Reference():
    def __init__(self, short, full):
        self.short = short
        self.full = full

    def __repr__(self):
        return self.full

    def __str__(self):
        return self.short


bispectrum = Reference(
    short='Bispectrum',
    full='Bispectrum.')

sigl = Reference(
    short='[Sigl & al 1994]',
    full='Sigl, J. C., & Chamoun, N. G. (1994). An introduction to bispectral '
    'analysis for the electroencephalogram. Journal of Clinical Monitoring '
    'and Computing, 10(6), 392-404.')

hagihira = Reference(
    short='[Hagihira & al 2001]',
    full='Hagihira, S., Takashina, M., Mori, T., Mashimo, T., & Yoshiya, I. '
    '(2001). Practical issues in bispectral analysis of '
    'electroencephalographic signals. Anesthesia & Analgesia, 93(4), 966-970.')

nagashima = Reference(
    short='[Nagashima & al 2006]',
    full='Nagashima, Y., Itoh, K., Itoh, S. I., Hoshino, K., Fujisawa, A., '
    'Ejiri, A., ... & Kusama, Y. (2006). Observation of coherent bicoherence '
    'and biphase in potential fluctuations around geodesic acoustic mode '
    'frequency on JFT-2M. Plasma physics and controlled fusion, 48(5A), A377.')

canolty = Reference(
    short='[Canolty & al 2006]',
    full='Canolty, R. T., Edwards, E., Dalal, S. S., Soltani, M., Nagarajan, '
    'S. S., Kirsch, H. E., ... & Knight, R. T. (2006). High gamma power is '
    'phase-locked to theta oscillations in human neocortex. science, '
    '313(5793), 1626-1628.')

penny = Reference(
    short='[Penny & al 2008]',
    full='Penny, W. D., Duzel, E., Miller, K. J., & Ojemann, J. G. (2008). '
    'Testing for nested oscillation. Journal of neuroscience methods, 174(1), '
    '50-61.')

colgin = Reference(
    short='[Colgin & al 2009]',
    full='Colgin, L. L., Denninger, T., Fyhn, M., Hafting, T., Bonnevie, '
    'T., Jensen, O., ... & Moser, E. I. (2009). Frequency of gamma '
    'oscillations routes flow of information in the hippocampus. '
    'Nature, 462(7271), 353-357.')

tort = Reference(
    short='[Tort & al 2010]',
    full='Tort, A. B., Komorowski, R., Eichenbaum, H., & Kopell, N. (2010). '
    'Measuring phase-amplitude coupling between neuronal oscillations of '
    'different frequencies. Journal of neurophysiology, 104(2), 1195-1210.')

ozkurt = Reference(
    short='[Ozkurt & al 2011]',
    full='Ozkurt, T. E., & Schnitzler, A. (2011). A critical note on the '
    'definition of phase-amplitude cross-frequency coupling. '
    'Journal of neuroscience methods, 201(2), 438-443.')

vanwijk = Reference(
    short='[van Wijk & al 2015]',
    full='van Wijk, B. C. M., Jha, A., Penny, W., & Litvak, V. (2015). '
    'Parametric estimation of cross-frequency coupling. '
    'Journal of neuroscience methods, 243, 94-102.')

jiang = Reference(
    short='[Jiang & al 2016]',
    full='Jiang, H., Bahramisharif, A., van Gerven, MA., & Jensen, O. (2015). '
    'Measuring directionality between neuronal oscillations of different '
    'frequencies. Neuroimage, 118, 359-367.')

duprelatour = Reference(
    short='[Dupre la Tour & al 2017]',
    full='Dupre la Tour, T. , Grenier, Y., & Gramfort, A. (2017). Parametric '
    'estimation of spectrum driven by an exogenous signal. Acoustics, Speech '
    'and Signal Processing (ICASSP), 2017 IEEE International Conference on,'
    '4301--4305.')

REFERENCES = {
    'bispectrum': bispectrum,
    'sigl': sigl,
    'hagihira': hagihira,
    'canolty': canolty,
    'nagashima': nagashima,
    'penny': penny,
    'colgin': colgin,
    'tort': tort,
    'ozkurt': ozkurt,
    'vanwijk': vanwijk,
    'jiang': jiang,
    'duprelatour': duprelatour,
}
