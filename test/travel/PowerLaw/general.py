import numpy as np

def prob_density_func(xs,norm=True):
    distKeys = set(xs)
    pdf = dict([(k,0) for k in distKeys])
    for x in xs:
        pdf[x] += 1
    if norm:
        pdf.update([(k,nf) for (k,nf) in zip(pdf.keys(),[float(f)/len(xs) for f in pdf.values()])])
    return pdf

def cum_density_func(xs,norm=True):
    pdf = prob_density_func(xs,norm)
    pdfk = pdf.keys()
    pdfk.sort()
    pdfv = map(pdf.get,pdfk)
    cdfv = np.cumsum(pdfv)
    return dict(zip(pdfk,cdfv))

def comp_cum_distribution(xs,norm=True):
    """Return the complement cumulative distribution(CCD) of a list dist

    Returns the CCD given by
            P(X>x) = 1 - P(X=<x)
    where P(X<x) is the cumulative distribution func given by
            P(X <= x) = sum_{xi<x} p(xi)
    where p(xi) is the probaility density func calculated as
            p(xi) = xi/(sum_i xi)

    Parameters
    ----------
    dist : list of Numeric
           list of values representing the frequency of a value at
           the index occuring
    norm : Whether or not to normalize the values by the sum

    Returns
    -------
    ccdh : dict of floats, keyed by occurance
           A dict of the same length, as the cumulative complementary
           distribution func.
    """
    cdf = cum_density_func(xs,norm)
    return dict(zip(cdf.keys(),[max(cdf.values()) - x for x in cdf.values()]))

def KSStat(S,P,reweight=False):
    """ Return the Kolomogorov-Smirnov statistic given two cumulative
    density functions, S of observed data, and P of the model.

    Returns the Kolomogorov-Smirnov statistic, defined as

        "    D=max|S(x)-P(x)|
        here S(x) is the CDF[cumulative distribution function] of the data
        for the observations...and P(x) is the CDF for the... model that
        bests fits the data..."

    On the matter of reweighting

        " The KS statistic is, for instance, known to be relatively
        insensitive to to differences between distributions at the extreme
        limits of the range of x because in these limits the CDFs necessarily
        tend to zero and one 

    Parameters
    ----------
    S : list of numeric
        cumulative distribution function of observed values
    P : list of numeric
        cumulative distribtuion function of model.
    """
    if reweight:
        return np.max([np.abs(s-p)/np.sqrt(p*(1.0-p)) for (s,p) in zip(S,P)])
    else:
        return np.max([np.abs(s-p) for (s,p) in zip(S,P)])
