package dataHandling;

import Share.Comm;
import Share.log;
import weka.core.Instance;
import weka.core.Instances;

public class InstancesHelper {

	
	/**
	 * get the train and test data for the given index fold.
	 * @param instances
	 * @param numFold:10
	 * @param index: 1..10
	 * @return Instances[] 0-test, 1-train
	 * @throws Exception 
	 */
//	public static Instances[] getFoldInstances(Instances MajorInsts, Instances MinorInsts, int numFold, int index) throws Exception
//	{		
//		DataInsts[] lstMajInsts = InstancesHelper.splitTrainTest(MajorInsts, numFold, index);
//		DataInsts[] lstMinInsts = InstancesHelper.splitTrainTest(MinorInsts, numFold, index);
//		
//		// Test
//		lstMajInsts[0].minorStart = lstMajInsts[0].size();
//		lstMajInsts[0].addAll(lstMinInsts[0]);
//		
//		// Train
//		lstMajInsts[1].minorStart = lstMajInsts[1].size();
//		lstMajInsts[1].addAll(lstMinInsts[1]);
//		return lstMajInsts;
//	}


	/**
	 * split instances into train data and test data.
	 * @param instances
	 * @param numFold
	 * @param index
	 * @return
	 * @throws Exception 
	 */
//	public static DataInsts[] splitTrainTest(Instances instances, int numFold, int index) throws Exception
//	{
//		DataInsts[] lstInsts = new DataInsts[2];
//		for(int i = 0; i < lstInsts.length; i++)
//		{
//			lstInsts[i] = new DataInsts(instances, 0);
//		}
//		
//		int sampleSize = instances.size() / numFold;
//
//		int beginIdx = sampleSize * index;
//		int endIdx   = sampleSize * (index + 1);
//		int idx = 0;
//		for(Instance instance: instances)
//		{
//			idx++;
//			if(idx >= beginIdx && idx < endIdx)
//			{
//				lstInsts[0].add(instance); // test data, 1 piece of fold.
//			}
//			else
//			{
//				lstInsts[1].add(instance); // train data, other 9 piece of fold.
//			}
//		}
//		
//		return lstInsts;
//	}
	
	/**
	 * subtract the list of instances
	 * @param instances
	 * @param beginIdx
	 * @param endIdx
	 * @return
	 */
	public static Instances drawSubList(Instances instances, int beginIdx, int endIdx)
	{
		Instances insts = new Instances(instances, 0);
		for(int i = beginIdx; i < endIdx; i++)
		{
			insts.add(instances.get(i));
		}
		
		return insts;
	}
	

	public static Instances getMajorInsts(Instances data) throws Exception
	{
		Instances instances = new Instances(data, 0);
		for(Instance instance : data)
		{
			if ((int)instance.classValue() == Comm.MAJOR_CLASS)
			{
				instances.add(instance);
			}
		}
		return instances;
	}	
	
	
	public static Instances getMinorInsts(Instances data) throws Exception
	{
		Instances instances = new Instances(data, 0);
		for(Instance instance : data)
		{
			if ((int)instance.classValue() == Comm.MINOR_CLASS)
			{
				instances.add(instance);
			}
		}
		return instances;
	}
}
