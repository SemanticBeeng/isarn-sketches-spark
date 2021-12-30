package util.distributionExtensions


import util.distributionExtensions.distributions._

/**
 *
 */
object syntax {

	// TODO try to make all the types work with syntax -- DiscrDist[D] as well as Dist[T, D] -- why not working?
	//  Need to put P[_,_] kind of thing?

	implicit class CDFSyntax[T: Numeric, D](distTD: Dist[T, D])(implicit ev: CDF[T, Dist[T, D]]){

		def cdf(x: T): Double = ev.cumProb(distTD, x)

		def inverseCdf(p: Double): T = ev.invCumProb(distTD, p)
	}


	implicit class DiscreteCDFSyntax[D](distpd: DiscreteDist[D])(implicit ev: CDF[IntZ, DiscreteDist[D]]){

		def cdf(x: IntZ): Double = ev.cumProb(distpd, x)
		def inverseCdf(p: Double): IntZ = ev.invCumProb(distpd, p)
	}

	/*implicit class DiscreteCDFSyntax[T: Numeric, D, P[_] <: Dist[T, D]](distpd: P[D])(implicit ev: CDF[T, P[D]]){

		def cdf(x: T): Double = ev.cumulativeProbability(distpd, x)
		//ev.cumulativeProbability(ev.getDistFromCDFArea(current), x)
		def inverseCdf(p: Double): T = ev.inverseCumulativeProbability(distpd, p)
	}*/

	implicit class ContinuousCDFSyntax[D](distTD: ContinuousDist[D])(implicit ev: CDF[Real, ContinuousDist[D]]){

		def cdf(x: Real): Double = ev.cumProb(distTD, x)
		//ev.cumulativeProbability(ev.getDistFromCDFArea(current), x)
		def inverseCdf(p: Double): Real = ev.invCumProb(distTD, p)
	}

	implicit class SamplingSyntax[T: Numeric, D](distTD: Dist[T, D])(implicit ev: Sampling[T, Dist[T, D]]){

		def sample(n: Int): Seq[T] = ev.sampleDist(distTD, n)
		def sample: T = ev.sampleDist(distTD, 1).head

	}

}
