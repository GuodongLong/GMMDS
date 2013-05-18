package dimReduction;

import java.text.DecimalFormat;

import Share.Comm;
import Share.log;
import weka.core.Matrix;


public class MDS {

	public Matrix getMDS(Matrix P, Matrix PoP) throws Exception
	{
		// mat = -0.5 * (I - 1/n * 1 * 1') * (P * P) * (I - 1/n * 1 * 1')
		//     = -0.5 * SC * CC * SC
		// ,where F = (-1/n * 1 * 1'), SC = (I - F'), CC = P * P
		int n = P.numRows();
        Matrix SC = getSizeComponent(n); // Side Component
        Matrix CC = PoP;          // Centre Component

		double[][] v = new double[n][n];
		double[] d = new double[n];
		double[] e = new double[n];
		
        Matrix mat = getIdentityMatrix(n, -0.5);
//		mat.eigenvalueDecomposition(v, d);
        mat = mat.multiply(SC);
//		mat.eigenvalueDecomposition(v, d);
        mat = mat.multiply(CC);
//		mat.eigenvalueDecomposition(v, d);//
        mat = mat.multiply(SC);
//        log.print("mat=\n" + mat.toString());
		checkSymmetric(mat);
		mat.eigenvalueDecomposition(v, d);
		
		v = sort(d, v);
		d = select(d);
		v = select(v);

		log.print(d);
//		log.print(v);
        // decomposition
		
		// get S = v * sqrt(d);
		Matrix V = new Matrix(v);
		Matrix D = getSqrtDiagMatrix(d);
		
		Matrix S = V.multiply(D);
		return S;
	}
	
	
	/**
	 * select the top N from vector
	 * @param d
	 * @return
	 */
	public double[] select(double [] d)
	{
		int n = d.length;
		int m = Comm.MAX_DIMENSION;
		double[] ds = new double[m];
		for(int i = 0; i < m; i++)
		{
			ds[i] = d[i];
		}
		return ds;
	}

	/**
	 * select the top N from vector
	 * @param v
	 * @return
	 */
	public double[][] select(double[][] v)
	{
		int n = v.length;
		int m = Comm.MAX_DIMENSION;
		double[][] vs = new double[n][m];
		for(int i = 0; i < n; i++)
		{
			for(int j = 0; j < m; j++)
			{
				vs[i][j] = v[i][j];
			}
		}
		return vs;
	}
	/**
	 * sort the eigenvalue and eigenvector based on the eigenvalue from max to min
	 * e.g.
	 * 21.557315878802083 1.4628432243943725 0.5000000000000007 0.5000000000000001 1.534470860258096E-15 -3.62760127246538E-16 -0.1056726707733531 -0.2704399392449873 -0.7551576042892336 
	 * @param d
	 * @param v
	 * @return
	 */
	public double[][] sort(double[] d, double[][] v)
	{
		int n = d.length;
		int[] ids = new int[n];
		
		for(int i = 0; i < n; i++)
		{
			double maxVal = -10000000;
			int maxId = 0;
			for(int j = i; j < n; j++)
			{
				if (d[j] > maxVal)
				{
					maxVal = d[j];
					maxId = j;
				}
			}
			d[maxId] = d[i];
			d[i] = maxVal;
			ids[i] = maxId;
		}
		
		// map transform from id to eigen vector
		double[][] vs = new double[n][n];
		for(int i = 0; i < n; i++)
		{
			for(int j = 0; j < n; j++)
			{
				double tmp = v[i][j];
				int newj = ids[j];
				vs[i][j] = v[i][newj];
			}
		}
		
		return vs;
	}
	
	public void checkSymmetric(Matrix mat)
	{
		int n = mat.numRows();
		for(int i = 0; i < n; i++)
		{
			for(int j = 0; j < i; j++)
			{
				mat.setElement(j, i, mat.getElement(i, j));
//				double d1 = mat.getElement(i, j);
//				double d2 = mat.getElement(j, i);
//				if (d1 != d2 && Math.abs(d1 - d2) < 0.00001)
//				{
//					double avg = (d1 + d2)/2;
//					mat.setElement(i, j, avg);
//					mat.setElement(j, i, avg);
////					log.print("i=" + i + ", j=" + j + ", ("+ mat.getElement(i, j) + ", " + mat.getElement(j, i) + ")");
//				}
			}
		}
		log.print("check Symmetric is ok!");
	}
	
	/**
	 * I - 1/n (1*1')
	 * @param nr
	 * @param nc
	 * @return
	 * @throws Exception
	 */
	public Matrix getSizeComponent(int n) throws Exception
	{
		Matrix I = getIdentityMatrix(n, 1);
		double dv = (double)-1/n;
		Matrix F = getFullMatrix(n, dv);
		Matrix C = I.add(F);
		return C;
	}
	
	/**
	 * Identity matrix
	 * 1 0 0 0 0 
	 * 0 1 0 0 0
	 * 0 0 1 0 0 
	 * 0 0 0 1 0
	 * 0 0 0 0 1
	 * 
	 * @param n
	 * @return
	 * @throws Exception
	 */
	public Matrix getIdentityMatrix(int n, double defaultValue) throws Exception
	{
		double[][] data = new double[n][n];
		for(int i = 0; i < n; i++)
		{
			data[i][i] = defaultValue;
		}
		Matrix mat = new Matrix(data);
		return mat;
	}
	
	/**
	 * Full matrix 
	 * -1 -1 -1 -1 -1
	 * -1 -1 -1 -1 -1
	 * -1 -1 -1 -1 -1
	 * -1 -1 -1 -1 -1
	 * -1 -1 -1 -1 -1
	 * 
	 * @param nr
	 * @param nc
	 * @return
	 * @t
	 */
	public Matrix getFullMatrix(int n, double defaultValue) throws Exception
	{
		double[][] data = new double[n][n];
		for(int i = 0; i < n; i++)
		{
			for(int j = 0; j < n; j++)
			{
				data[i][j] = defaultValue;
			}
		}
		Matrix mat = new Matrix(data);
		return mat;
	}

	/**
	 * Get a diagonal matrix with (4, 16, 9, 1)
	 * 4 0  0 0
	 * 0 16 0 0
	 * 0 0  9 0
	 * 0 0  0 1
	 * @param n
	 * @param defaultValue
	 * @return
	 * @throws Exception
	 */
	public Matrix getDiagMatrix(int n, double[] defaultValue) throws Exception
	{
		double[][] data = new double[n][n];
		for(int i = 0; i < n; i++)
		{
			data[i][i] = defaultValue[i];
		}
		Matrix mat = new Matrix(data);
		return mat;
	}
	
	/**
	 * Get a sqrt diagonal matrix with (4, 16, 9, 1)
	 * 2 0 0 0
	 * 0 4 0 0
	 * 0 0 2 0
	 * 0 0 0 1
	 * @param n
	 * @param defaultValue
	 * @return
	 * @throws Exception
	 */
	public Matrix getSqrtDiagMatrix(double[] d) throws Exception
	{
		int n = d.length;
		double[][] data = new double[n][n];
		for(int i = 0; i < n; i++)
		{
			data[i][i] = Math.sqrt(d[i]);
		}
		Matrix mat = new Matrix(data);
		return mat;
	}
}
