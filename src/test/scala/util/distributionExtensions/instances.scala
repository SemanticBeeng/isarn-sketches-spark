package util.distributionExtensions

import util.distributionExtensions.distributions._
/**
 *
 */
object instances {

	trait PoissonInstances {

		implicit def poissonDistHasCDF: CDF[IntZ, Dist[IntZ, PoissonDist]] =
			new CDF[IntZ, Dist[IntZ, PoissonDist]] {

				def cumProb(d: Dist[IntZ, PoissonDist], n: IntZ): Double = {
					d.getDist.cumulativeProbability(n.intValue())
				}
				def invCumProb(d: Dist[IntZ, PoissonDist], p: Double): IntZ = {
					BigInt(d.getDist.inverseCumulativeProbability(p))
				}
			}
		implicit def poissonDiscreteDistHasCDF: CDF[IntZ, DiscreteDist[PoissonDist]] =
			new CDF[IntZ, DiscreteDist[PoissonDist]] {

				def cumProb(d: DiscreteDist[PoissonDist], n: IntZ): Double = {
					d.getDist.cumulativeProbability(n.intValue())
				}
				def invCumProb(d: DiscreteDist[PoissonDist], p: Double): IntZ = {
					BigInt(d.getDist.inverseCumulativeProbability(p))
				}
			}
		// TODO: try to avoid this repetition - try to declare CDF[T, D, P[_,_]]?
		// MOTIVation: the shortrange-longrange functions in kolmogorovtryout
		/*implicit def poissonDiscreteDistHasCDF[P[_] <: Dist[Z, PoissonDist]]: CDF[Z, P[PoissonDist]] = new
				CDF[Z, P[PoissonDist]] {

			def cumulativeProbability(d: P[PoissonDist], n: Z): Double = {
				d.getDist.cumulativeProbability(n)
			}
			def inverseCumulativeProbability(d: P[PoissonDist], p: Double): Z = {
				d.getDist.inverseCumulativeProbability(p)
			}
		}*/
		/*implicit def poissonDistHasCDF: CDF[Z, PoissonDist] = new CDF[Z, PoissonDist] {

			def cumulativeProbability(d: Dist[Z, PoissonDist], n: Z): Double = {
				d.getDist.cumulativeProbability(n)
			}
			def inverseCumulativeProbability(d: Dist[Z, PoissonDist], p: Double): Z = {
				d.getDist.inverseCumulativeProbability(p)
			}
		}*/

		implicit def poissonDistCanBeSampled: Sampling[IntZ, Dist[IntZ, PoissonDist]] =
			new Sampling[IntZ, Dist[IntZ, PoissonDist]]{

				def sampleDist(d: Dist[IntZ, PoissonDist], n: Int): Seq[IntZ] =
					d.getDist.sample(n.intValue()).map(BigInt(_))
			}
	}
	//import PoissonInstances._

	trait BinomialInstances {
		implicit def binomialDistHCDF: CDF[IntZ, Dist[IntZ, BinomialDist]] =
			new CDF[IntZ, Dist[IntZ, BinomialDist]] {

				def cumProb(d: Dist[IntZ, BinomialDist], n: IntZ): Double = {
					d.getDist.cumulativeProbability(n.intValue())
				}
				def invCumProb(d: Dist[IntZ, BinomialDist], p: Double): IntZ = {
					BigInt(d.getDist.inverseCumulativeProbability(p))
				}
			}
		implicit def binomialDistCanBeSampled: Sampling[IntZ, Dist[IntZ, BinomialDist]] =
			new Sampling[IntZ, Dist[IntZ, BinomialDist]]{

				def sampleDist(d: Dist[IntZ, BinomialDist], n: Int): Seq[IntZ] =
					d.getDist.sample(n.intValue()).map(BigInt(_))
			}
	}

	trait GeometricInstances {
		implicit def geometricDistHCDF: CDF[IntZ, Dist[IntZ, GeometricDist]] =
			new CDF[IntZ, Dist[IntZ, GeometricDist]] {

				def cumProb(d: Dist[IntZ, GeometricDist], n: IntZ): Double = {
					d.getDist.cumulativeProbability(n.intValue())
				}
				def invCumProb(d: Dist[IntZ, GeometricDist], p: Double): IntZ = {
					BigInt(d.getDist.inverseCumulativeProbability(p))
				}
			}

		implicit def geometricDistCanBeSampled: Sampling[IntZ, Dist[IntZ, GeometricDist]] =
			new Sampling[IntZ, Dist[IntZ, GeometricDist]]{

				def sampleDist(d: Dist[IntZ, GeometricDist], n: Int): Seq[IntZ] =
					d.getDist.sample(n).map(BigInt(_))
			}

	}

	trait GammaInstances {
		implicit def gammaDistHasCDF: CDF[Real, Dist[Real, GammaDist]] =
			new CDF[Real, Dist[Real, GammaDist]] {

				def cumProb(d: Dist[Real, GammaDist], x: Real): Double = {
					d.getDist.cumulativeProbability(x.doubleValue())
				}
				def invCumProb(d: Dist[Real, GammaDist], p: Double): Real = {
					BigDecimal.valueOf(d.getDist.inverseCumulativeProbability(p))
				}
			}
		implicit def gammaDistCanBeSampled: Sampling[Real, Dist[Real, GammaDist]] =
			new Sampling[Real, Dist[Real, GammaDist]]{

				def sampleDist(d: Dist[Real, GammaDist], n: Int): Seq[Real] =
					d.getDist.sample(n).map(BigDecimal.valueOf(_))
			}

	}

	trait NormalInstances {
		implicit def normalDistHasCDF: CDF[Real, Dist[Real, NormalDist]] = new CDF[Real, Dist[Real, NormalDist]] {

			def cumProb(d: Dist[Real, NormalDist], x: Real): Double = {
				d.getDist.cumulativeProbability(x.doubleValue())
			}
			def invCumProb(d: Dist[Real, NormalDist], p: Double): Real = {
				BigDecimal.valueOf(d.getDist.inverseCumulativeProbability(p))
			}
		}
		implicit def normalDistCanBeSampled: Sampling[Real, Dist[Real, NormalDist]] =
			new Sampling[Real, Dist[Real, NormalDist]]{

				def sampleDist(d: Dist[Real, NormalDist], n: Int): Seq[Real] =
					d.getDist.sample(n).map(BigDecimal.valueOf(_))
			}
	}

	trait ContinuousUniformInstances {
		implicit def uniformContinuousDistHasCDF: CDF[Real, Dist[Real, ContinuousUniformDist]] = new
				CDF[Real, Dist[Real, ContinuousUniformDist]] {
			def cumProb(d: Dist[Real, ContinuousUniformDist], x: Real): Double = {
				d.getDist.cumulativeProbability(x.doubleValue())
			}
			def invCumProb(d: Dist[Real, ContinuousUniformDist], p: Double): Real = {
				val ic = d.getDist.inverseCumulativeProbability(p)
				println(s"INV CUM PROB NUMBER FORMAT EXCEPTION: $ic")
				BigDecimal.valueOf(ic)
			}
		}
		implicit def continuousUniformDistCanBeSampled: Sampling[Real, Dist[Real, ContinuousUniformDist]] =
			new Sampling[Real, Dist[Real, ContinuousUniformDist]]{

				def sampleDist(d: Dist[Real, ContinuousUniformDist], n: Int): Seq[Real] =
					d.getDist.sample(n).map(BigDecimal.valueOf(_))
			}
	}

	trait GumbelInstances {
		implicit def gumbelDistHasCDF: CDF[Real, Dist[Real, GumbelDist]] = new CDF[Real, Dist[Real,
			GumbelDist]] {
			def cumProb(d: Dist[Real, GumbelDist], x: Real): Double = {
				d.getDist.cumulativeProbability(x.doubleValue())
			}
			def invCumProb(d: Dist[Real, GumbelDist], p: Double): Real = {
				BigDecimal.valueOf(d.getDist.inverseCumulativeProbability(p))
			}
		}
		implicit def gumbelDistCanBeSampled: Sampling[Real, Dist[Real, GumbelDist]] = new Sampling[Real,
			Dist[Real, GumbelDist]]{
			def sampleDist(d: Dist[Real, GumbelDist], n: Int): Seq[Real] =
				d.getDist.sample(n).map(BigDecimal.valueOf(_))
		}
	}

	trait ExponentialInstances {
		implicit def exponentialDistHasCDF: CDF[Real, Dist[Real, ExponentialDist]] = new CDF[Real, Dist[Real,
			ExponentialDist]] {
			def cumProb(d: Dist[Real, ExponentialDist], x: Real): Double = {
				d.getDist.cumulativeProbability(x.doubleValue())
			}
			def invCumProb(d: Dist[Real, ExponentialDist], p: Double): Real = {
				BigDecimal.valueOf(d.getDist.inverseCumulativeProbability(p))
			}
		}
		implicit def exponentialDistCanBeSampled: Sampling[Real, Dist[Real, ExponentialDist]] = new Sampling[Real,
			Dist[Real, ExponentialDist]]{
			def sampleDist(d: Dist[Real, ExponentialDist], n: Int): Seq[Real] =
				d.getDist.sample(n).map(BigDecimal.valueOf(_))
		}
	}


	trait DiscreteCDFInstances extends PoissonInstances
		with GeometricInstances
		with BinomialInstances


	trait ContinuousCDFInstances extends NormalInstances
		with GammaInstances
		with ContinuousUniformInstances
		with GumbelInstances

	object AllInstances extends DiscreteCDFInstances with ContinuousCDFInstances

}
