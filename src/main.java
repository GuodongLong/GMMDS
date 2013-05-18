

import graph.Graph;
import graph.graphHelper;

import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

import libsvm.svm;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.clusterers.AbstractClusterer;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Matrix;
import weka.core.converters.ConverterUtils.DataSource;
import dataHandling.DataLoader;
import dataHandling.MDSMatrix;
import dataHandling.Sparse;
import dataHandling.matrixHelper;
import dimReduction.MDS;
import Exception.DataFileNotFind;
import Exception.DataFileReadError;
import Exception.DataSourceLoadError;
import Exception.IllegalEnumValue;
import Exception.TimerException;
import Share.Comm;
import Share.FileOption;
import Share.log;

public class main {

//	public static String dataset = "NRM";
	public static String dataset = "20NG";
	
	public static void main(String[] args) throws Exception
	{
//		testCase();
//	   realCase(Comm.NRMFileName);
//		realCase_small(FileOption.CS_IN_DOMAIN);
		for(int i = 1; i <= 1; i++)
		{
			log.print("Weight of MDS = " + i);
			realCase_all(FileOption.CS_IN_DOMAIN, i);
		}
//		for(int i = 1; i <= 50; i++)
//		{
//			log.print("Weight of MDS = " + i);
//			realCase_sparse(FileOption.CS_IN_DOMAIN, i);
//		}
	}
	
	
	public static void testCase() throws Exception
	{

		double[][] PT = {
				{0,1,1,1,2,2,3,3,4},
				{1,0,1,2,3,3,4,4,5},
				{1,1,0,1,2,2,3,3,4},
				{1,2,1,0,1,1,2,2,3},
				{2,3,2,1,0,1,1,1,2},
				{2,3,2,1,1,0,1,1,2},
				{3,4,3,2,1,1,0,1,1},
				{3,4,3,2,1,1,1,0,2},
				{4,5,4,3,2,2,1,2,0}
		};
		double[][] PoPdata = matrixHelper.getPoP(PT);
		
		// using MDS to reduce dimensions
		Matrix P = new Matrix(PT);
		Matrix PoP = new Matrix(PoPdata);
		log.print("P=\n" + P.toString());
		MDS mds = new MDS();
		Matrix S = mds.getMDS(P, PoP);
//		log.print(S.toString());
	}
	
	public static void realCase_all(String filename, int WEIGHT_MDS) throws Exception
	{
		int numFold = 10;
		// get a matrix for proximity
		Instances instances = DataLoader.loadDataFromFile(filename);
		instances.randomize(Comm.random);
		Instances train = instances.testCV(numFold, 0);
		Instances test = instances.trainCV(numFold, 0);
/*		
		Graph graph = new Graph();
		graphHelper.init(graph, instances); // build graph with the given instances data.
		
		MDSMatrix mdsMat = graphHelper.buildMatrix(graph);
		double[][] PoPdata = matrixHelper.getPoP(mdsMat.data);
		
		String matDataFile = Comm.dataPath + dataset + "/mat_all.data";
		log.print("start to save matrix file to data file " + matDataFile);
		matrixHelper.saveMatrixToFile(mdsMat, matDataFile);
		
		// using MDS to reduce dimensions
		log.print("start to do MDS");
		Matrix P = new Matrix(mdsMat.data);
		Matrix PoP = new Matrix(PoPdata);
		MDS mds = new MDS();
		Matrix S = mds.getMDS(P, PoP);
//		log.print("\n" + matrixToString(S, train));
		
		// clustering and improve classification
		String matWekaFile = Comm.dataPath + dataset + "/matWekafile_all.arff";
		matrixToWekaFile(S, train, matWekaFile); // no class value;
		log.print("saved matrix file to weka file " + matWekaFile);		
		
		String matLabFile = Comm.dataPath + dataset + "/matlabfile_all.data";
		saveMatlabFile(S, instances, matLabFile);
		log.print("saved matrix file to matlab file " + matLabFile);
*/

		String matWekaFile = Comm.dataPath + dataset + "/matWekafile_all.arff";
		String clusterResultFile = "";
		DataSource source = new DataSource(matWekaFile);
		Instances clusterTrainData = source.getDataSet();
		AbstractClusterer cluster = clustering(clusterTrainData, clusterResultFile);
		
		Attribute attr_classid = new Attribute("classId");
		Instances clusterTrainDataLab = new Instances(clusterTrainData);
		clusterTrainDataLab.insertAttributeAt(attr_classid, clusterTrainDataLab.numAttributes()); 
		clusterTrainDataLab.setClassIndex(clusterTrainDataLab.numAttributes()-1);
		for(int i = 0; i < clusterTrainDataLab.numInstances(); i++)
		{
			clusterTrainDataLab.get(i).setClassValue(instances.get(i).classValue());
		}
		String matWekaFileLab = Comm.dataPath + dataset + "/matWekafile_all_lab.arff";
		saveInstsToWekaFile(clusterTrainDataLab, matWekaFileLab);
		
		// attach clusterId to each instance
		// insert attributes
		Instances newInsts = new Instances(instances);
		Instances clusterTestData = new Instances(clusterTrainData);
		Attribute attr_clusterid = new Attribute("clusterId");
		clusterTestData.insertAttributeAt(attr_clusterid, clusterTestData.numAttributes()); // it has not label, so the insert position is different

		for(int k = 0; k < WEIGHT_MDS; k++)
		{
			for(int c = 0; c < clusterTestData.numAttributes(); c++)
			{
				Attribute attribute = new Attribute("MDS_attr_" + c + "_" + k);
				newInsts.insertAttributeAt(attribute, newInsts.numAttributes()-1); // it has label, so the insert position is different
			}
		}
		// insert values
		for(int i = 0; i < clusterTrainData.size(); i++)
		{
			int clusterValue = cluster.clusterInstance(clusterTrainData.get(i));
			clusterTestData.get(i).setValue(clusterTestData.numAttributes() - 1, clusterValue);
			for(int k = 0; k < WEIGHT_MDS; k++)
			{
				for(int c = 0; c < clusterTestData.numAttributes(); c++)
				{
					double value = clusterTestData.get(i).value(c);
					newInsts.get(i).setValue(instances.numAttributes() - 1 + k*clusterTestData.numAttributes() + c, value);
				}
			}
		}
		newInsts.setClassIndex(newInsts.numAttributes() - 1);
		

		String augWekaFile = Comm.dataPath + dataset + "/augWekafile_all_w" + WEIGHT_MDS + ".arff";
		saveInstsToWekaFile(newInsts, augWekaFile);
		
//		log.print(clusterTestData.toString());
		
		// help classifier
		Instances classifierTrainData1 = new Instances(train);
//		NaiveBayes cl1 = new NaiveBayes();
		weka.classifiers.functions.LibSVM cl1 = new weka.classifiers.functions.LibSVM();
		cl1.setOptions(getOptionsForSVM());
		cl1.buildClassifier(classifierTrainData1);
		Evaluation eval1 = new Evaluation(test);
		eval1.evaluateModel(cl1, test);
		log.print(eval1.toSummaryString());

		// train classifier with the help of proximity data and cluster label.

		Instances train_mds = newInsts.testCV(numFold, 0);
		Instances test_mds = newInsts.trainCV(numFold, 0);
//		NaiveBayes cl2 = new NaiveBayes();
		weka.classifiers.functions.LibSVM cl2 = new weka.classifiers.functions.LibSVM();
		cl2.setOptions(getOptionsForSVM());
		cl2.buildClassifier(train_mds);
		Evaluation eval2 = new Evaluation(test_mds);
		eval2.evaluateModel(cl2, test_mds);
		log.print(eval2.toSummaryString());
	}

	public static void realCase_sparse(String filename, int WEIGHT_MDS) throws Exception
	{
		// get a matrix for proximity
		Instances instances = DataLoader.loadDataFromFile(filename);
		instances.randomize(Comm.random);
		instances = Sparse.removeValues(instances);
		Instances train = instances.testCV(10, 0);
		Instances test = instances.trainCV(10, 0);
/*	
		Graph graph = new Graph();
		graphHelper.init(graph, instances); // build graph with the given instances data.
		
		MDSMatrix mdsMat = graphHelper.buildMatrix(graph);
		double[][] PoPdata = matrixHelper.getPoP(mdsMat.data);
		
		String matDataFile = Comm.dataPath + dataset + "/mat_all.data";
		log.print("start to save matrix file to data file " + matDataFile);
		matrixHelper.saveMatrixToFile(mdsMat, matDataFile);
		
		// using MDS to reduce dimensions
		log.print("start to do MDS");
		Matrix P = new Matrix(mdsMat.data);
		Matrix PoP = new Matrix(PoPdata);
		MDS mds = new MDS();
		Matrix S = mds.getMDS(P, PoP);
//		log.print("\n" + matrixToString(S, train));
		
		// clustering and improve classification
		String matWekaFile = Comm.dataPath + dataset + "/matWekafile_sparse.arff";
		matrixToWekaFile(S, matWekaFile); // no class value;
		log.print("saved matrix file to weka file " + matWekaFile);		
		
		String matLabFile = Comm.dataPath + dataset + "/matlabfile_sparse.data";
		saveMatlabFile(S, instances, matLabFile);
		log.print("saved matrix file to matlab file " + matLabFile);
*/

		String matWekaFile = Comm.dataPath + dataset + "/matWekafile_sparse.arff";
		log.print(matWekaFile);
		String clusterResultFile = "";
		DataSource source = new DataSource(matWekaFile);
		Instances clusterTrainData = source.getDataSet();
		AbstractClusterer cluster = clustering(clusterTrainData, clusterResultFile);
		
		// attach clusterId to each instance
		// insert attributes
		Instances newInsts = new Instances(instances);
		Instances clusterTestData = new Instances(clusterTrainData);
		Attribute attr_clusterid = new Attribute("clusterId");
		clusterTestData.insertAttributeAt(attr_clusterid, clusterTestData.numAttributes()); // it has not label, so the insert position is different
		
		for(int k = 0; k < WEIGHT_MDS; k++)
		{
			for(int c = 0; c < clusterTestData.numAttributes(); c++)
			{
				Attribute attribute = new Attribute("MDS_attr_" + c + "_" + k);
				newInsts.insertAttributeAt(attribute, newInsts.numAttributes()-1); // it has label, so the insert position is different
			}
		}
		// insert values
		for(int i = 0; i < clusterTrainData.size(); i++)
		{
			int clusterValue = cluster.clusterInstance(clusterTrainData.get(i));
			clusterTestData.get(i).setValue(clusterTestData.numAttributes() - 1, clusterValue);
			for(int k = 0; k < WEIGHT_MDS; k++)
			{
				for(int c = 0; c < clusterTestData.numAttributes(); c++)
				{
					double value = clusterTestData.get(i).value(c);
					newInsts.get(i).setValue(instances.numAttributes() - 1 + k*clusterTestData.numAttributes() + c, value);
				}
			}
		}

		newInsts.setClassIndex(newInsts.numAttributes() - 1);
		String augWekaFile = Comm.dataPath + dataset + "/augWekafile_all_w" + WEIGHT_MDS + ".arff";
		saveInstsToWekaFile(newInsts, augWekaFile);
		
//		log.print(clusterTestData.toString());
		
		// help classifier
/*		Instances classifierTrainData1 = new Instances(train);
		NaiveBayes nb1 = new NaiveBayes();
		nb1.buildClassifier(classifierTrainData1);
		Evaluation eval1 = new Evaluation(test);
		eval1.evaluateModel(nb1, test);
		log.print(eval1.toSummaryString());
*/
		// train classifier with the help of proximity data and cluster label.

		Instances train_mds = newInsts.testCV(10, 0);
		Instances test_mds = newInsts.trainCV(10, 0);
		NaiveBayes nb2 = new NaiveBayes();
		nb2.buildClassifier(train_mds);
		Evaluation eval2 = new Evaluation(test_mds);
		eval2.evaluateModel(nb2, test_mds);
		log.print(eval2.toSummaryString());
	}
	
	/**
	 * Save the instances object into a weka file
	 * @param instances
	 * @param filename
	 * @throws IOException
	 */
	public static void saveInstsToWekaFile(Instances instances, String filename) throws IOException
	{
		FileWriter fw = new FileWriter(filename);
		fw.write(instances.toString());
		fw.close();
	}
	
	/**
	 * Save the matrix file to visualise the distribution on Matlab.
	 * @param S
	 * @param instances
	 * @param filename
	 * @throws IOException
	 */
	public static void saveMatlabFile(Matrix S, Instances instances, String filename) throws IOException
	{
		StringBuilder sb = new StringBuilder();
		int n = S.numRows();
		int m = S.numColumns();
		for(int i = 0; i < n; i++)
		{
			for(int j = 0; j < m; j++)
			{
				sb.append(S.getElement(i, j) + ",");
			}
			sb.append(instances.get(i).classValue() + "\n");
		}

		FileWriter fw = new FileWriter(filename);	
		fw.write(sb.toString());
		fw.close();
	}
	
	/**
	 * Transform the matrix object to weka file.
	 * @param S
	 * @param instances
	 * @param filename
	 * @throws IOException
	 */
	public static void matrixToWekaFile(Matrix S, String filename) throws IOException
	{
		FileWriter fw = new FileWriter(filename);

		StringBuilder sb = new StringBuilder();
		
		// title & attribute
		int n = S.numRows();
		int m = S.numColumns(); 
		sb.append("@relation MDS_Matrix\n\n");
		for(int i = 0; i < m; i++)
		{
			sb.append("@attribute attr_" + i + " real\n");
		}
		
		// classes
		/*
		int nc = instances.numClasses()-1;
		sb.append("\n@attribute class {");
		for(int c = 0; c < nc; c++)
		{
			sb.append(c + ".0,");
		}
		sb.append(nc + ".0}\n");
		*/
		
		// data
		sb.append("\n@data\n");
		DecimalFormat df = new DecimalFormat("#.##");
		for(int i = 0; i < n; i++)
		{
			for(int j = 0; j < m-1; j++)
			{
				sb.append(df.format(S.getElement(i, j)) + ",");
			}
			sb.append(df.format(S.getElement(i, m-1)) + "\n");
//			sb.append(instances.get(i).classValue() + "\n");
		}
		fw.write(sb.toString());
		fw.close();
	}

	/**
	 * Convert a matrix object to a String
	 * @param S
	 * @return
	 */
	public static String matrixToString(Matrix S, Instances instances)
	{
		StringBuilder sb = new StringBuilder();
		
		int n = S.numRows();
		int m = S.numColumns(); 
		DecimalFormat df = new DecimalFormat("#.##");
		for(int i = 0; i < n; i++)
		{
			for(int j = 0; j < m; j++)
			{
				sb.append(df.format(S.getElement(i, j)) + ",");
			}
			sb.append(instances.get(i).classValue() + "\n");
		}
		
		return sb.toString();
	}
	
	/**
	 * Do clustering on the low-dimension data
	 * @param instances
	 * @param output
	 * @return
	 * @throws Exception
	 */
	public static AbstractClusterer clustering(Instances instances, String output) throws Exception
	{		
		// remove the class value from training data
		Instances trainData = new Instances(instances);
		
		// train a new cluster
		AbstractClusterer cluster = buildSimpleKMeans(trainData);
		
		return cluster;
		// evaluate the cluster
//	    ClusterEvaluation eval = new ClusterEvaluation();
//	    eval.setClusterer(cluster);
//	    eval.evaluateClusterer(instances);
//		FileWriter fw_cluster = new FileWriter(output);
//		fw_cluster.write(eval.clusterResultsToString());
//		fw_cluster.close();
	}
	
	/**
	 * Build a KMeans cluster
	 * @param instances
	 * @return
	 * @throws Exception
	 */
	public static AbstractClusterer buildSimpleKMeans(Instances instances) throws Exception
	{
		String[] options = new String[4];
		options[0] = "-N";
		options[1] = "2";
		options[2] = "-M";
		options[3] = "10000";
		String algName = "KMeans"+options[1];
		
		AbstractClusterer cluster = new SimpleKMeans();
		((SimpleKMeans) cluster).setOptions(options);
		cluster.buildClusterer(instances);
		return cluster;
	}
	
	/**
	 * Get the options for SVM classifier
	 * @return
	 */
	public static String[] getOptionsForSVM()
	{
		String opt = "-S 1 -K 2 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.0010 -P 0.1";
		String[] options = opt.split(" ");
		return options;
	}
}
