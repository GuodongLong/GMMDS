package dataHandling;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;


import Share.log;

public class matrixHelper {

	/**
	 * Save a matrix object into a file.
	 * @param mat
	 * @param fileName
	 * @throws IOException
	 */
	public static void saveMatrixToFile(MDSMatrix mat, String fileName) throws IOException
	{
		FileWriter fw = new FileWriter(fileName);
		
		fw.write(mat.numLine + " " + mat.numCol + "\n");
		
		for(int i = 0; i < mat.numLine; i++)
		{
			StringBuilder sb = new StringBuilder();
			for(int j = 0; j < mat.numLine; j++)
			{
				sb.append(mat.data[i][j] + " ");
			}
			fw.write(sb.toString());
			fw.write("\n");
		}
		fw.close();
		
		log.print("Write " + mat.numLine + " X " + mat.numCol + " matrix into file " + fileName);
	}
	

	/**
	 * Get the element-wise matrix multiplication
	 * AoB[i][j] = A{i][j] * B[[i][j] 
	 * or
	 * PoP[i][j] = P{i][j] * P[[i][j] 
	 * @param P
	 * @return
	 */
	public static double[][] getPoP(double[][] P)
	{
		int n = P.length;
		double[][] PoP = new double[n][n];
		
		for(int i = 0; i < n; i++)
		{
			for(int j = 0; j < n; j++)
			{
				double p = P[i][j];
				PoP[i][j] = p * p;
			}
		}
		
		return PoP;
	}
	
	/**
	 * Get a matrix object from a file.
	 * @param fileName
	 * @return
	 * @throws IOException
	 */
	public static MDSMatrix readMatrixFromFile(String fileName) throws IOException
	{
		FileReader fr = new FileReader(fileName);
		BufferedReader br = new BufferedReader(fr);
		
		String line = br.readLine();
		String[] str = line.split(" ");
		int numLine = Integer.valueOf(str[0]);
		int numCol = Integer.valueOf(str[1]);
		
		MDSMatrix mat = new MDSMatrix(numLine, numCol);
		int ln = 0;
		while((line = br.readLine()) != null)
		{
			str = line.split(" ");
			for(int col = 0; col < numCol; col++)
			{
				mat.data[ln][col] = Integer.valueOf(str[col]);
			}
			ln++;
		}
		
		br.close();
		fr.close();
		log.print("Read " + mat.numLine + "X" + mat.numCol + " matrix from file " + fileName);
		return mat;
	}
	
	/**
	 *  
	 * @param mat1
	 * @param mat2
	 * @return (mat1-mat2)
	 */
	public static MDSMatrix minus(MDSMatrix mat1, MDSMatrix mat2)
	{
		MDSMatrix mat = new MDSMatrix(mat1.numLine, mat1.numCol);
		
		for(int i = 0; i < mat1.numLine; i++)
		{
			for(int j = 0; j < mat1.numCol; j++)
			{
				mat.data[i][j] = mat1.data[i][j] - mat2.data[i][j];
			}
		}
		
		return mat;
	}
	

	/**
	 * multiple two matrix
	 * @param mat1
	 * @param mat2
	 * @return (mat1*mat2)
	 */
	public static MDSMatrix multiple(MDSMatrix mat1, MDSMatrix mat2)
	{
		MDSMatrix mat = new MDSMatrix(mat1.numLine, mat2.numCol);
		
		for(int i = 0; i < mat1.numLine; i++)
		{
			for(int j = 0; j < mat1.numCol; j++)
			{
				mat.data[i][j] = multiple(mat1.getLine(i), mat2.getColumn(j));
			}
		}
		
		return mat;
	}
	
	/**
	 * multiple two vectors or arrays
	 * @param v1
	 * @param v2
	 * @return |v1*v2|
	 */
	public static double multiple(double[] v1, double[] v2)
	{
		double ret = 0;
		for(int i = 0; i < v1.length; i++)
		{
			for(int j = 0; j < v2.length; j++)
			{
				ret += v1[i] * v2[j];
			}
		}
		return ret;
	}
	
}
