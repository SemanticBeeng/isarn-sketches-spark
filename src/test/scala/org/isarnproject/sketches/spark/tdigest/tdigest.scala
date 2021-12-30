/*
Copyright 2017-2020 Erik Erlandson
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package org.isarnproject.sketches.spark.tdigest

import org.apache.commons.math3.analysis.UnivariateFunction
import org.apache.commons.math3.analysis.integration.{SimpsonIntegrator, UnivariateIntegrator}
import org.apache.spark.ml.linalg.{Vectors => VectorsML}
import org.apache.spark.mllib.linalg.{Vectors => VectorsMLLib}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row}
import org.isarnproject.sketches.java.TDigest
import org.isarnproject.testing.spark.SparkTestSuite
import utest._

import scala.math.pow
import org.apache.commons.math3.analysis.function.Exp
import org.apache.commons.math3.special.Gamma

/*import org.apache.commons.math3.distribution.{GammaDistribution, GeometricDistribution, GumbelDistribution,
	PoissonDistribution}*/
import util.distributionExtensions.distributions._
import util.distributionExtensions.instances.AllInstances._
import util.distributionExtensions.syntax._


import scala.reflect.runtime.universe._
import util.GeneralUtil

import scala.util.Random._

object TestData {
	var TEST_ID: Int = 1
	var KSD_COUNT: Int = 1

	val MAX_DISCRETE: Int = 500 // effort to make discrete dist tdigests longer!
}

import TestData._


object TDigestAggregationSuite extends SparkTestSuite {



	import CDFFunctions._

	// set the seed before generating any data
	setSeed(7337L * 3773L)

	val NUM_TDIGEST_SEQS: Int = 10

	// don't use lazy values because then data generation order may be undefined,
	// due to test execution order
	/*val data1: DataFrame = spark.createDataFrame(Vector.fill(10001){(nextInt(10), nextGaussian)})
		.toDF("j","x")
		.cache()*/
	val data1: DataFrame = spark.createDataFrame(Vector.fill(10001){
		(ContinuousUniformDist(0, 9).sample, NormalDist(0,1).sample)})
		.toDF("j","x")
		.cache()

	val data2: DataFrame = spark.createDataFrame(
		Vector.fill(10002){(
			ContinuousUniformDist(0, 9).sample,
			Vector.fill(NUM_TDIGEST_SEQS){NormalDist(0,1).sample})})
		.toDF("j", "x")
		.cache()

	val data3: DataFrame = spark.createDataFrame(
		Vector.fill(10003){(
			ContinuousUniformDist(0, 9).sample,
			VectorsML.dense(NormalDist(0,1).sample(NUM_TDIGEST_SEQS)))})
		.toDF("j", "x")
		.cache()

	val data4: DataFrame = spark.createDataFrame(
		Vector.fill(10004){(
			ContinuousUniformDist(0, 9).sample,
			VectorsMLLib.dense(NormalDist(0,1).sample(NUM_TDIGEST_SEQS)))})
		.toDF("j", "x")
		.cache()

	val GAMMA_SHAPE: Double = 2.0
	val GAMMA_SCALE: Double = 3.0
	val gammaData: DataFrame = spark.createDataFrame(
		Vector.fill(10004){(
			ContinuousUniformDist(0, 9).sample,
			GammaDist(GAMMA_SHAPE, GAMMA_SCALE).sample(NUM_TDIGEST_SEQS))})
		.toDF("j", "x")
		.cache()


	val POISSON_RATE: Double = 7.8
	val poissonData: DataFrame = spark.createDataFrame(
		Vector.fill(10004){(
			ContinuousUniformDist(0, 9).sample,
			PoissonDist(POISSON_RATE).sample(NUM_TDIGEST_SEQS))})
		.toDF("j", "x")
		.cache()

	val GUMBEL_MU: Double = 5.0
	val GUMBEL_SIGMA: Double = 3.0
	val gumbelExtremeValueData: DataFrame = spark.createDataFrame(
		Vector.fill(10004){(
			ContinuousUniformDist(0, 9).sample,
			GumbelDist(GUMBEL_MU, GUMBEL_SIGMA).sample(NUM_TDIGEST_SEQS))})
		.toDF("j", "x")
		.cache()


	val GEOM_P: Double = 0.32
	val geometricData: DataFrame = spark.createDataFrame(
		Vector.fill(10004){(
			ContinuousUniformDist(0, 9).sample,
			GeometricDist(GEOM_P).sample(NUM_TDIGEST_SEQS))})
		.toDF("j", "x")
		.cache()


	// Spark DataFrames and RDDs are lazy.
	// Make sure data are actually created prior to testing, or ordering
	// may change based on test ordering
	val count1: Long = data1.count()
	val count2: Long = data2.count()
	val count3: Long = data3.count()
	val count4: Long = data4.count()
	val gammaCount: Long = gammaData.count()
	val poissonCount: Long = poissonData.count()
	val gumbelCount: Long = gumbelExtremeValueData.count()
	val geometricCount: Long = geometricData.count()


	// Kolmogorov-Smirnov D tolerance
	val epsD = 0.022 // NOTE: changed from 0.02 because continuous uniform test gave 0.02019 once
	// TODO how to find out what is the tolerance for gaussian, uniform, gamma, multimodal etc?


	val tests = Tests {
		test("TDigestAggregator") {
			assert(data1.rdd.partitions.size > 1)

			val udf: UserDefinedFunction = TDigestAggregator.udf[Double](compression = 0.25, maxDiscrete = MAX_DISCRETE)

			val agg: Row = data1.agg(udf(col("j")), udf(col("x"))).first

			val (tdj, tdx): (TDigest, TDigest) = (agg.getAs[TDigest](0), agg.getAs[TDigest](1))

			approx(tdj.mass(), count1)
			approx(tdx.mass(), count1)

			// Set the count for the test
			TEST_ID = 1
			val ksdU = KSD(tdj, ContinuousUniformDist(0,9))
			val ksdN = KSD(tdx, NormalDist(0,1))

			assert(ksdU < epsD)
			assert(ksdN < epsD)
		}


		test("TDigestArrayAggregator") {
			assert(data2.rdd.partitions.size > 1)

			val udfj: UserDefinedFunction = TDigestAggregator.udf[Double](maxDiscrete = MAX_DISCRETE)
			val udfx: UserDefinedFunction = TDigestArrayAggregator.udf[Double](compression = 0.25)

			val agg: Row = data2.agg(udfj(col("j")), udfx(col("x"))).first

			val (tdj, tdx): (TDigest, Seq[TDigest]) = (agg.getAs[TDigest](0), agg.getAs[Seq[TDigest]](1))

			approx(tdj.mass(), count2)

			// Set the count ID for the test
			TEST_ID = 2

			val ksdU = KSD(tdj, ContinuousUniformDist(0,9))
			assert(ksdU < epsD)

			var numTD: Int = 1
			for { td <- tdx } {
				approx(td.mass(), count2)

				val ksdN = KSD(td, NormalDist(0,1))
				assert(ksdN < epsD)

				numTD += 1
			}
		}

		test("TDigestArrayAggregator - gamma data"){
			assert(gammaData.rdd.partitions.size > 1) // TODO check it has to do with the uniform-gamma tuple split

			val udfj: UserDefinedFunction = TDigestAggregator.udf[Double](maxDiscrete = MAX_DISCRETE)
			val udfx: UserDefinedFunction = TDigestArrayAggregator.udf[Double](compression = 0.25)

			val agg: Row = gammaData.agg(udfj(col("j")), udfx(col("x"))).first

			val (tdj, tdx): (TDigest, Seq[TDigest]) = (agg.getAs[TDigest](0), agg.getAs[Seq[TDigest]](1))

			approx(tdj.mass(), gammaCount)

			TEST_ID = 3

			val ksdU = KSD(tdj, ContinuousUniformDist(0, 9))
			assert(ksdU < epsD)


			// TODO need to alter the epsilon? error rate for skewed distributions? (check open tabs for different
			//  rules for skewed distribution check)
			var numTD = 1
			for { td <- tdx } {
				approx(td.mass(), gammaCount)

				val ksd = KSD(td, GammaDist(GAMMA_SHAPE, GAMMA_SCALE))
				numTD += 1

				assert(ksd < epsD)
			}
		}

		test("TDigestArrayAggregator - poisson data"){
			assert(poissonData.rdd.partitions.size > 1) // TODO check it has to do with the uniform-gamma tuple split

			val udfj: UserDefinedFunction = TDigestAggregator.udf[Double](maxDiscrete = MAX_DISCRETE)
			// NOTE setting max discrete here to be higher is the KEY to making the t-digest estimate converge!!!
			val udfx: UserDefinedFunction = TDigestArrayAggregator.udf[Int](compression = 0.25, maxDiscrete = MAX_DISCRETE)

			val agg: Row = poissonData.agg(udfj(col("j")), udfx(col("x"))).first

			val (tdj, tdx): (TDigest, Seq[TDigest]) = (agg.getAs[TDigest](0), agg.getAs[Seq[TDigest]](1))

			approx(tdj.mass(), poissonCount)

			TEST_ID = 4

			val ksdU = KSD(tdj, ContinuousUniformDist(0, 9))
			assert(ksdU < epsD)


			// TODO need to alter the epsilon? error rate for skewed distributions? (check open tabs for different
			//  rules for skewed distribution check)

			// Sequence of t-digests getting each compared with singular cdf (comparing multiple t-digests with
			// multiple cdfs)
			var numTD = 1
			for { td <- tdx } {
				approx(td.mass(), poissonCount)
				val ksd: Double = KSD(td, PoissonDist(POISSON_RATE))

				numTD += 1
				assert(ksd < epsD)
			}

		}


		test("TDigestArrayAggregator - geometric data"){
			assert(geometricData.rdd.partitions.size > 1) // TODO check it has to do with the uniform-gamma tuple
			// split

			val udfj: UserDefinedFunction = TDigestAggregator.udf[Double](maxDiscrete = 25)
			// NOTE change to Int here for poisson
			val udfx: UserDefinedFunction = TDigestArrayAggregator.udf[Int](compression = 0.25, maxDiscrete =
				MAX_DISCRETE)

			val agg: Row = geometricData.agg(udfj(col("j")), udfx(col("x"))).first

			val (tdj, tdx): (TDigest, Seq[TDigest]) = (agg.getAs[TDigest](0), agg.getAs[Seq[TDigest]](1))

			TEST_ID = 5
			approx(tdj.mass(), geometricCount)
			val ksdU = KSD(tdj, ContinuousUniformDist(0, 9))
			assert(KSD(tdj, ContinuousUniformDist(0, 9)) < epsD)


			// TODO need to alter the epsilon? error rate for skewed distributions? (check open tabs for different
			//  rules for skewed distribution check)

			// Sequence of t-digests getting each compared with singular cdf (comparing multiple t-digests with
			// multiple cdfs)
			var count = 1
			for { td <- tdx } {
				approx(td.mass(), geometricCount)
				val ksd: Double = KSD(td, GeometricDist(GEOM_P))

				count += 1
				assert(ksd < epsD)
			}

		}

		test("TDigestArrayAggregator - gumbel extreme data"){
			assert(gumbelExtremeValueData.rdd.partitions.size > 1) // TODO check it has to do with the uniform-gamma
			// tuple split

			val udfj: UserDefinedFunction = TDigestAggregator.udf[Double](maxDiscrete = 25)
			val udfx: UserDefinedFunction = TDigestArrayAggregator.udf[Double](compression = 0.25)

			val agg: Row = gumbelExtremeValueData.agg(udfj(col("j")), udfx(col("x"))).first

			val (tdj, tdx): (TDigest, Seq[TDigest]) = (agg.getAs[TDigest](0), agg.getAs[Seq[TDigest]](1))


			TEST_ID = 6

			approx(tdj.mass(), gumbelCount)

			val ksdU = KSD(tdj, ContinuousUniformDist(0, 9))
			assert(ksdU < epsD)


			// TODO need to alter the epsilon? error rate for skewed distributions? (check open tabs for different
			//  rules for skewed distribution check)
			for { td <- tdx } {
				approx(td.mass(), gumbelCount)

				val ksdG = KSD(td, GumbelDist(GUMBEL_MU, GUMBEL_SIGMA))
				assert(ksdG < epsD)
			}
		}

		/*test("TDigestMLVecAggregator") {
			assert(data3.rdd.partitions.size > 1)

			val udfj: UserDefinedFunction = TDigestAggregator.udf[Int](maxDiscrete = 25)
			val udfx: UserDefinedFunction = TDigestMLVecAggregator.udf(compression = 0.25)
			val agg: Row = data3.agg(udfj(col("j")), udfx(col("x"))).first
			val (tdj, tdx): (TDigest, Seq[TDigest]) = (agg.getAs[TDigest](0), agg.getAs[Seq[TDigest]](1))
			approx(tdj.mass(), count3)
			assert(KSD(tdj, ContinuousUniformDist(0, 9)) < epsD)

			for { td <- tdx } {
				approx(td.mass(), count3)
				assert(KSD(td, NormalDist(0, 1)) < epsD)
			}
		}

		test("TDigestMLLibVecAggregator") {
			assert(data4.rdd.partitions.size > 1)

			val udfj: UserDefinedFunction = TDigestAggregator.udf[Int](maxDiscrete = 25)
			val udfx: UserDefinedFunction = TDigestMLLibVecAggregator.udf(compression = 0.25)
			val agg: Row = data4.agg(udfj(col("j")), udfx(col("x"))).first
			val (tdj, tdx): (TDigest, Seq[TDigest]) = (agg.getAs[TDigest](0), agg.getAs[Seq[TDigest]](1))
			approx(tdj.mass(), count4)
			assert(KSD(tdj, ContinuousUniformDist(0, 9)) < epsD)

			for { td <- tdx } {
				approx(td.mass(), count4)
				assert(KSD(td, NormalDist(0, 1)) < epsD)
			}
		}

		test("TDigestReduceAggregator") {
			assert(data1.rdd.partitions.size > 1)

			val udf: UserDefinedFunction = TDigestAggregator.udf[Double](compression = 0.25)
			val grp: DataFrame = data1.groupBy("j").agg(udf(col("x")).alias("td"))
			assert(grp.count() == 10)

			val udfred: UserDefinedFunction = TDigestReduceAggregator.udf(compression = 0.25)
			val agg: Row = grp.agg(udfred(col("td"))).first
			val tdx: TDigest = agg.getAs[TDigest](0)
			approx(tdx.mass(), count1)

			assert(KSD(tdx, NormalDist(0,1)) < epsD)
		}

		test("TDigestArrayReduceAggregator") {
			assert(data2.rdd.partitions.size > 1)

			val udf: UserDefinedFunction = TDigestArrayAggregator.udf[Double](compression = 0.25)
			val grp: DataFrame = data2.groupBy("j").agg(udf(col("x")).alias("td"))
			assert(grp.count() == 10)

			val udfred: UserDefinedFunction = TDigestArrayReduceAggregator.udf(compression = 0.25)
			val agg: Row = grp.agg(udfred(col("td"))).first
			val tdx: Seq[TDigest] = agg.getAs[Seq[TDigest]](0)

			for { td <- tdx } {
				approx(td.mass(), count2)
				assert(KSD(td, NormalDist(0, 1)) < epsD)
			}
		}*/
	}
}

object CDFFunctions {


	/*type ContinuousCDF = Double => Double
	type DiscreteCDF = Int => Double*/


	// Kolmogorov Smirnov D-statistic

	def KSD[T: Numeric : TypeTag, D](tdgst: TDigest, dist: Dist[T, D], n: Int = 10000)(implicit evCdf: CDF[T, Dist[T,
		D]])
	: Double	= {
		require(tdgst.size() > 1)
		require(n > 0)


		/*val xmin: T = dist.inverseCdf(0) // TODO
		val xmax: T = typeOf[T].toString.split('.').last match {
			case "IntZ" => BigInt(1000000).asInstanceOf[T]
			case "Real" => BigDecimal.valueOf(100000.0).asInstanceOf[T]
		}*/
		//val xmax: T = dist.inverseCdf(1) // TODO weird error BigDecimal numberformatException here -- why? too big?
		val xmin: Double = tdgst.cdfInverse(0) // x-value at beginning (total area = 0)
		// NOTE: trying to make longer t-digest (longer x-range maybe?) seem to be directly related
		val xmax: Double = tdgst.cdfInverse(1) //10000.0


		val xvals: Seq[T] = GeneralUtil.generateTSeqFromDouble[T](xmin, xmax)
		//val xvals: Seq[T] = GeneralUtil.generateTSeq[T](xmin, xmax)

		val tdCdf: T => Double = d => {
			val dd: Double = new java.lang.Double(implicitly[Numeric[T]].toDouble(d))

			if (tdgst.size() <= tdgst.getMaxDiscrete()) tdgst.cdfDiscrete(dd) else tdgst.cdf(dd)
		}


		// Calculates the KSD statistic number here:
		val ksd: Double  = xvals
			.iterator
			.map(x => math.abs(tdCdf(x) - dist.cdf(x)))
			.max

		println("----------------------------------------------------------------------------------------")
		println(s"Test #$TEST_ID")
		println(s"Distribution = ${dist.toString} |  typeOf[T] = ${typeOf[T].toString.split('.').last}")
		println(s"xmin = $xmin, xmax = $xmax)  |  xvals = [${xvals.take(5)}, ..., ${xvals.drop(xvals.length - 5)}]")
		println(s"xvals.length = ${xvals.length}")
		println(s"tdigest size = ${tdgst.size()}")
		println(s"ksd #$KSD_COUNT = $ksd")
		println("----------------------------------------------------------------------------------------")

		ksd
	}






	/*def gaussianCDF(mean: Double = 0, stdv: Double = 1): ContinuousCDF = {
		require(stdv > 0.0)

		val z = stdv * math.sqrt(2.0)
		(x: Double) => (1.0 + erf((x - mean) / z)) / 2.0
	}

	def discreteUniformCDF(xmin: Double, xmax: Double): ContinuousCDF = {
		require(xmax > xmin)
		require(xmin >= 0)

		val z = (1 + xmax - xmin).toDouble

		(x: Double) => {
			if (x < xmin.toDouble) 0.0 else if (x >= xmax.toDouble) 1.0 else {
				val xi = x.toInt
				(1 + xi - xmin).toDouble / z
			}
		}
	}


	// https://en.wikipedia.org/wiki/Error_function#Approximation_with_elementary_functions
	def erf(x: Double): Double = {
		// erf is an odd function
		if (x < 0.0) -erf(-x) else {
			val t = 1.0 / (1.0 + (0.47047 * x))
			var u = t
			var s = 0.0
			s += 0.3480242 * u
			u *= t
			s -= 0.0958798 * u
			u *= t
			s += 0.7478556 * u
			s *= math.exp(-(x * x))
			1.0 - s
		}
	}

	// TODO gamma cdf, poisson cdf, etc
	// TODO update wolfram project by documenting with the proofs of the CDFs (how to derive the CDFs)

	// Source = https://statproofbook.github.io/D/gam
	//WARNING using this function instead of gammaCDF (auto) results in "too many evaluations" error from apache
	// commons math.
	def manualGammaCDF(aShape: Double, bRate: Double): ContinuousCDF = {
		require(aShape > 0)
		require(bRate > 0)

		val e = new Exp()

		def integrandFunc: Double => UnivariateFunction = s => new UnivariateFunction() {
			def value(t: Double): Double = pow(t, s-1) * e.value(-t)
		}

		val uniIntg: UnivariateIntegrator = new SimpsonIntegrator()
		val incompleteGammaIntegral: (Double, Double) => Double =
			(s, x) => uniIntg.integrate(100, integrandFunc(s),0, x)

		val gammaCDFFunction: Double => Double = (x) =>
			incompleteGammaIntegral(aShape, bRate * x) / Gamma.gamma(aShape)

		gammaCDFFunction
		// TODO: check against apache gamma regularized computation: https://commons.apache.org/proper/commons-math/javadocs/api-3.0/org/apache/commons/math3/special/Gamma.html#regularizedGammaP(double,%20double)
	}


	def gammaCDF(aShape: Double, bRate: Double): ContinuousCDF = {
		require(aShape > 0)
		require(bRate > 0)

		(x: Double) => GammaDist(aShape, bRate).cdf(x)
	}

	def poissonCDF(rate: Double): DiscreteCDF = {
		require(rate > 0)

		// TODO: cutting off decimal part, but is floor the best way? or just use .toInt without floor?
		(n: Int) => PoissonDist(rate).cumulativeProbability(n)
	}

	def gumbelExtremeValueCDF(mu: Double, sigma: Double): ContinuousCDF = {
		require(sigma > 0)

		(x: Double) => GumbelDist(mu, sigma).cumulativeProbability(x)
	}

	def geometricCDF(probSuccess: Double): DiscreteCDF = {
		require(probSuccess >= 0)

		(n: Int) => GeometricDist(probSuccess).cumulativeProbability(n)
	}*/
}
