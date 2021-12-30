package util.distributionExtensions

/**
 *
 */



import org.apache.commons.math3.distribution.{BinomialDistribution, ExponentialDistribution, GammaDistribution, GeometricDistribution, GumbelDistribution, NormalDistribution, PoissonDistribution, UniformRealDistribution}


/**
 * GOAL: have a supertype Distribution[T] and also the functionality of all distributions (cdf, pdf, sample, ...)
 */
object distributions {

	type IntZ = BigInt // integers
	type Real = BigDecimal // real numbers


	trait CDF[T, D] {
		//def getDistFromCDFArea(d: D): D // NOTE: this functions not necessary for the experiment of making
		// AbsDist[T] callable with .cdf
		def cumProb(d: D, x: T): Double
		def invCumProb(d: D, p: Double): T
	}
	trait Sampling[T, D] {
		def sampleDist(d: D, n: Int): Seq[T]
	}
	trait Dist[T, D] /*extends CDF[T, AbsDist[T, D]]*/{
		def getDist: D
	}
	trait ContinuousDist[D] extends Dist[Real, D]
	trait DiscreteDist[D] extends Dist[IntZ, D]


	case class PoissonDist(lambda: Double) extends PoissonDistribution(lambda)	with DiscreteDist[PoissonDist]
	{ def getDist: PoissonDist = this }
	case class BinomialDist(numTrials: Int, p: Double) extends BinomialDistribution(numTrials, p)
		with DiscreteDist[BinomialDist] { def getDist: BinomialDist = this }
	case class GeometricDist(p: Double) extends GeometricDistribution(p)
		with DiscreteDist[GeometricDist] { def getDist: GeometricDist = this }

	case class GammaDist(shape: Double, scale: Double) extends GammaDistribution(shape, scale)
		with ContinuousDist[GammaDist] { def getDist: GammaDist = this }
	case class NormalDist(mu: Double, std: Double) extends NormalDistribution(mu, std)
		with ContinuousDist[NormalDist] { def getDist: NormalDist = this }
	case class ContinuousUniformDist(a: Double, b: Double) extends UniformRealDistribution(a, b)
		with ContinuousDist[ContinuousUniformDist] { def getDist: ContinuousUniformDist = this }
	case class GumbelDist(mu: Double, beta: Double) extends GumbelDistribution(mu, beta)
		with ContinuousDist[GumbelDist] { def getDist: GumbelDist = this }
	case class ExponentialDist(mean: Double) extends ExponentialDistribution(mean)
		with ContinuousDist[ExponentialDist] { def getDist: ExponentialDist = this }


}