package Share;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class Comm {

	public static String workPath = "/home/gulong/workspace/MDS/";
	public static String logPath  = workPath + "log/";
	public static String dataPath = workPath + "data/";
	public static String tempPath = workPath + "temp/";

//	public static String workPath = "E:\\don\\TTAN\\";
//	public static String logPath  = workPath + "log\\";
//	public static String dataPath = workPath + "data\\";
//	public static String tempPath = workPath + "temp\\";
	
	public static String NRMFileName = FileOption.NRM_file;
	public static String WikiFileName = FileOption.Wiki_file; 
	public static String UnlabeledFileName = dataPath + "unlabeled_wiki_Weka.arff";
	
	public static String WekaTemplate = dataPath + "template_Weka.txt";
	
	public static String TrainSetFileName = Comm.tempPath + "trainSet.arff";
	public static String TestSetFileName  = Comm.tempPath + "testSet.arff";
	
	public static double MIN_DOUBLE = 1e-75;
	public static double MIN_MUTUAL_INFO = 0.0652; //0.005406894170951938
	public static int    ATTR_SIZE = 6112;
	
	public static int MINOR_CLASS = 0;
	public static int MAJOR_CLASS = 1;
	
	public static int MAX_ITERATION = 15;
	

	// the minimal shared words count for building a new edge.
	public static int MIN_SharedWord = 1;
	
	// infinite number of the distance
	public static int INFINITE_DISTANCE = 100000;
	
	// dimensions of MDS
	public static int MAX_DIMENSION = 5;

//	public static String labeledFile  = FileOption.CS_IN_DOMAIN_Nominal;
//	public static String unlabeledFile = FileOption.CS_OUT_DOMAIN_Nominal;
	public static String labeledFile  = FileOption.Wiki_file;
	public static String unlabeledFile = FileOption.NRM_file;
	public static int numAttr = 0, numInst = 0, numClass = 0;
	public static double[] classRatio;
	public static int[] classDocSize;

	public static Instances D_l, D_u, D_u_l;
	public static Classifier tan_l = null;
	
	public static Random random = new Random(1);
	
	/**
	 * Array
	 */
	
	public static void clearArray(int[] ar)
	{
		for(int i = 0; i < ar.length; i++)
		{
			ar[i] = 0;
		}
	}

	/**
	 * Normalized the array.
	 * @param ar
	 */
	public static void NormalizeArray(double[] ar)
	{
		int iSum = 0;
		for(int i = 0; i < ar.length; i++)
		{
			iSum += ar[i];
		}
		for(int i = 0; i < ar.length; i++)
		{
			ar[i] = ar[i]/iSum;
		}
	}
	
	/**
	 * Double Format
	 */
	public static String formatDouble(double d)
	{
		return formatDouble(d, "#0.000000");
	}
	
	public static String formatDouble(double d, String str)
	{
		NumberFormat format = new DecimalFormat(str);
		return format.format(d);
	}
	
	/**
	 * Get the maximal value of the given array.
	 * @param arr
	 * @return
	 * @throws NullPointerException
	 */
	public static double max(double[] arr) throws NullPointerException
	{
		if (arr == null || arr.length < 1)
		{
			log.print("Error num of array.");
			throw new NullPointerException();
		}
		double maxVal = arr[0];
		for(double d: arr)
		{
			if (d > maxVal)
			{
				maxVal = d;
			}
		}
		return maxVal;
	}

	public static double KL(Instances insts1, Instances insts2)
	{
		double KL = 0;
		
		// estimate the probability for each word
		int numAttr = insts1.numAttributes() - 1;
		int D1 = 0, D2 = 0;
		
		int[] occurrence1 = new int[numAttr];
		for(Instance instance : insts1)
		{
			for(int i = 0; i < numAttr; i++)
			{
				if (instance.value(i) > 0)
				{
					occurrence1[i]++;
					D1++;
				}
			}
		}

		int[] occurrence2 = new int[numAttr];
		for(Instance instance : insts2)
		{
			for(int i = 0; i < numAttr; i++)
			{
				if (instance.value(i) > 0)
				{
					occurrence2[i]++;
					D2++;
				}
			}
		}

		double[] prob1 = new double[numAttr];
		double[] prob2 = new double[numAttr];
		for(int i = 0; i < numAttr; i++)
		{
			prob1[i] = (double)(1 + occurrence1[i]) / (numAttr + D1);
			prob2[i] = (double)(1 + occurrence2[i]) / (numAttr + D2);
		}
		
		// calculate the KL
		for(int i = 0; i < numAttr; i++)
		{
			KL += prob1[i] * Math.log(prob1[i] / prob2[i]) / Math.log(2);
		}
//		log.print("KL = " + KL);
		return KL;
	}
	
}
