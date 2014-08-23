/**
 * Prototype code for Kvasir project.
 *
 * Liang Wang @ CS Dept, Helsinki University, Finland
**/

import scala.util.Random
import breeze.linalg._
import breeze.numerics._
import org.netlib.util.intW
import com.github.fommil.netlib.LAPACK.{getInstance=>lapack}
import org.apache.spark.rdd._
import org.apache.spark.mllib._
import org.apache.spark.Partition
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._



class ApproxMatrix() {

    var allFeatures = 0
    var approxRank = 0
    var overSample = 100

    def loadMatrix(path: String) = {
    	val path = "/data/test_matrix"
	//val samples = 100
    	var rawMatrix = sc.textFile(path)
	val nRawRows = rawMatrix.count
	val nParts = (1.0 * nRawRows / samples).ceil.asInstanceOf[Int]
	rawMatrix = rawMatrix.repartition(nParts)
	val A = rawMatrix.mapPartitions(part => { part.map(line => SparseVector[Double](line.split(",").map(_.toDouble))) }, preservesPartitioning = true)
    }

    def reductionOnSubMatrix(part: Iterator[SparseVector[Double]], numRow: Int, numCol: Int) = {
    	var y = x.mapPartitions(part => {
     	    var partial = CSCMatrix.zeros[Double](numRow, numCol)
     	    part.map(vec => {
	        val rndVec = SparseVector(1.to(numCol).map(_ => Random.nextGaussian):_*)
		partial = partial + vec.t.t * rndVec.t
	        vec
	    } )
     	})
    }

    def reduceDimension(vecRDD: RDD[SparseVector[Double]]) = {
    	vecRDD.mapPartitions(reductionOnSubMatrix(_), preservesPartitioning = true)
    }

    def toSparseVectors(rawMatrix: RDD[String]) = {
    	rawMatrix.mapPartitions(part => { part.map(line => SparseVector[Double](line.split(",").map(_.toDouble))) }, preservesPartitioning = true)
    }

    def setApproxRank(rank: Int) = {
        approxRank = rank
    }

    /**
     *  Calculate the approx matrix given a fat-short matrix.
    **/
    def approxSVD(A: RDD[SparseVector[Double]], rank: Int, overSample: Int, powerInteration: Int) = {
    	var tY = A.map(vec => {
	    val rndVec = SparseVector(1.to(rank + overSample).map(_ => Random.nextGaussian):_*)
	    vec.t.t * rndVec.t
	}).reduce(_ + _)

	// test: Try aggregate
	var tX = A.aggregate[CSCMatrix[Double]](CSCMatrix.zeros[Double](100000,1200))(
	    seqOp = (s: CSCMatrix[Double], vec: SparseVector[Double]) => {
	    	val rndVec = SparseVector(1.to(rank + overSample).map(_ => Random.nextGaussian):_*)
            	s + vec.t.t * rndVec.t
	    },
	    combOp = (s1: CSCMatrix[Double], s2: CSCMatrix[Double]) => s1 + s2
	)

	val (q,r) = reducedQR(tY.toDense)

	for (i <- 0 until powerInteration) {
	    // todo
	}

	val b = matrixProduct(q.t, A)
	val c = covarianceMatrix(b)
	val (u,s,v) = svd(b)
	((q * u).apply(::,0 until rank), sqrt(s(0 until rank)), v)
    }

    /**
     *  Perform one power iteration, we divide Q matrix
     *  into columns to avoid memory issue by giving up
     *	some performance.
    **/
    def powerIteration () = {
    }

    /**
     *  Perform multiplication on two sparse matrix.
    **/
    def sparseMatrixProduct(A: CSCMatrix, B: CSCMatrix) = {
    	//todo
    }

    /**
     *  Perform multiplication between a dense matrix and
     *  a sparse RDD, and A << B. A is small and B is fat.
     *  The returned matrix is another fat-short matrix.
    **/
    def matrixProduct(A: DenseMatrix[Double], B: RDD[SparseVector[Double]]): RDD[DenseVector[Double]] = {
    	val bA = sc.broadcast(A)
	B.map(vec => bA.value * vec)
    }

    /**
     *  Perform multiplication between a sparse RDD and a
     *  dense matrix, and A >> B. A is tall and B is small.
     *  The returned matrix is small.
    **/
    def matrixProduct2(A: RDD[SparseVector[Double]], B: DenseMatrix[Double]): RDD[DenseVector[Double]] = {
    	val bA = sc.broadcast(A)
	B.map(vec => bA.value * vec)
    }

    /**
     *  Calculate covariance matrix A*A'. A is fat-short
     *  matrix stored in RDD. The resulted matrix is small.
    **/
    def covarianceMatrix(A: RDD[DenseVector[Double]]): DenseMatrix[Double] = {
    	A.map(vec => { vec * vec.t }).reduce(_ + _)
    }

    /**
     *  Convert a dense matrix to sparse one.
    **/
    def convertDense2Sparse(A: DenseMatrix[Double]):  = {
      //todo
    }

    /**
     *  A's shape is m,n, m must be bigger than n.
    **/
    def reducedQR(A: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {

	val m = A.rows
    	val n = A.cols

	// Get optimal workspace size
    	val scratch, work = new Array[Double](1)
    	val info = new intW(0)
    	lapack.dgeqrf(m, n, scratch, m, scratch, work, -1, info)
    	val lwork1 = if(info.`val` != 0) n else work(0).toInt
    	lapack.dorgqr(m, m, scala.math.min(m,n), scratch, m, scratch, work, -1, info)
    	val lwork2 = if(info.`val` != 0) n else work(0).toInt
    	val workspace = new Array[Double](scala.math.max(lwork1, lwork2))

    	//Perform the QR factorization with dgeqrf
    	val maxd = scala.math.max(m,n)
    	val mind = scala.math.min(m,n)
    	val tau = new Array[Double](mind)
    	val outputMat = DenseMatrix.zeros[Double](m,mind)
    	for(r <- 0 until m; c <- 0 until n)
      	      outputMat(r,c) = A(r,c)
    	lapack.dgeqrf(m, n, outputMat.data, m, tau, workspace, workspace.length, info)

    	//Error check
    	//if (info.`val` > 0)
      	//   throw new NotConvergedException(NotConvergedException.Iterations)
    	//else if (info.`val` < 0)
      	//     throw new IllegalArgumentException()

    	//Get R
    	val R = DenseMatrix.zeros[Double](n,n)
    	for(c <- 0 until maxd if(c < n); r <- 0 until m if(r <= c))
    	    R(r,c) = outputMat(r,c)

      	//Get Q from the matrix returned by dgep3
      	val Q = DenseMatrix.zeros[Double](m,mind)
      	lapack.dorgqr(m, mind, mind, outputMat.data, m, tau, workspace, workspace.length, info)
      	for(r <- 0 until m; c <- 0 until n)
            Q(r,c) = outputMat(r,c)

      	//Error check
      	//if (info.`val` > 0)
        //    throw new NotConvergedException(NotConvergedException.Iterations)
      	//else if (info.`val` < 0)
        //    throw new IllegalArgumentException()

      	(Q,R)
    }

}