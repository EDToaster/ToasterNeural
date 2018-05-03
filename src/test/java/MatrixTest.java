import org.junit.Test;
import util.math.Matrix;
import util.exception.NeuralNetworkMatrixSizeException;

public class MatrixTest {

	Matrix testMatrix0 = new Matrix(new double[][]{
			{1, 2},
			{3, 4},
			{5, 6}
	});

	Matrix testMatrix1 = new Matrix(new double[][]{
			{1, 2},
			{3, 4}
	});

	Matrix testTransposed0 = new Matrix(new double[][]{
			{1, 2, 3}
	});

	Matrix testTransposed1 = new Matrix(new double[][]{
			{4, 5, 6}
	});

	Matrix testVector0 = testTransposed0.transpose();
	Matrix testVector1 = testTransposed1.transpose();


	@Test
	public void testTranspose() {
		Matrix transposed = new Matrix(new double[][]{
				{1, 3, 5},
				{2, 4, 6}
		});

		assert testMatrix0.transpose().equals(transposed);
	}

	@Test
	public void testSplicing() {
		Matrix col_0 = new Matrix(new double[][]{
				{1},
				{3},
				{5}
		});

		Matrix col_t0 = new Matrix(new double[][]{
				{1},
				{2}
		});

		assert testMatrix0.spliceColumn(0).equals(col_0);
		assert testMatrix0.transpose().spliceColumn(0).equals(col_t0);
		assert testMatrix0.spliceRow(0).equals(col_t0);
		assert testMatrix0.transpose().spliceRow(0).equals(col_0);
	}

	@Test
	public void testIsVector() {
		assert testVector0.isVector();
		assert !testVector0.transpose().isVector();
		assert !testMatrix0.isVector();
		assert !testMatrix0.transpose().isVector();
		assert !testTransposed0.isVector();
		assert testTransposed0.transpose().isVector();
	}

	@Test
	public void testDoubleDotProduct() {
		assert testVector0.dot(testVector1) == 32;
	}

	@Test(expected = NeuralNetworkMatrixSizeException.class)
	public void testDoubleDotProductError() {
		testVector0.dot(testMatrix0);
	}

	@Test
	public void testSet() {
		Matrix result = new Matrix(new double[][]{{1, 2, 2}}).transpose();
		result.set(2, 0, 3);
		assert result.equals(testVector0);
	}

	@Test
	public void testMatrixMatrixMultiplication() {
		Matrix resultMatrix = new Matrix(new double[][]{
				{7, 10},
				{15, 22},
				{23, 34}
		});
		assert resultMatrix.equals(testMatrix1.mult(testMatrix0));
	}

	@Test
	public void testMatrixScaling(){
		Matrix resultMatrix = new Matrix(new double[][]{
				{2, 4},
				{6, 8}
		});
		assert resultMatrix.equals(testMatrix1.mult(2));
	}

	@Test
	public void testIdentity() {
		Matrix id = Matrix.identity(3);
		assert testMatrix0.mult(id).equals(testMatrix0);
	}

	@Test
	public void testAdding() {
		Matrix result = new Matrix(new double[][]{
				{2, 4},
				{6, 8}
		});

		assert result.equals(testMatrix1.add(testMatrix1));
	}

	@Test
	public void testSubbing() {
		Matrix result = new Matrix(new double[][]{
				{0,0},
				{0,0}
		});

		assert result.equals(testMatrix1.sub(testMatrix1));
	}

	@Test
	public void testCost() {
		assert testMatrix0.calculateDistanceCost(testMatrix0) == 0;

		Matrix broken = new Matrix(new double[][]{
				{0, 0},
				{3, 4},
				{5, 6}
		});

		assert testMatrix0.calculateDistanceCost(broken) == 5;
		assert broken.calculateDistanceCost(testMatrix0) == 5;
	}

}
