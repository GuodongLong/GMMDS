package dataHandling;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import Exception.DataFileNotFind;
import Exception.DataFileReadError;
import Exception.DataSourceLoadError;
import Exception.IllegalEnumValue;
import Exception.TimerException;
import Share.log;

public class Sparse {

	public static double SPARSE_RATE = 0.90;
	
	public static Instances imbalancData(Instances instances)
	{
		int numClass = instances.numClasses();
		int[] classId = new int[numClass];
		int[] classCnt = new int[numClass];
		int[] classCnt2 = new int[numClass];
		Instances[] instss = new Instances[numClass];
		
		Instances newInsts = new Instances(instances, 0);
		for(int i = 0; i < numClass; i++)
		{
			classId[i] = i;
			classCnt[i] = 0;
			instss[i] = new Instances(instances, 0);
		}

		for(Instance instance : instances)
		{
			int classVal = (int) instance.classValue();
			classCnt[classVal]++;
			instss[classVal].add(instance);
		}
		log.print(classCnt);
		
		for(int i = 0; i < numClass; i++)
		{
			int leftCnt = (int) (classCnt[0] * 1/(i*2+1));
			double leftRatio = (double)leftCnt / classCnt[i];
			log.print(leftRatio);
			Random random = new Random(1);
			for(Instance instance : instss[i])
			{
				if (random.nextDouble() <= leftRatio)
				{
					newInsts.add(instance);
				}
			}
		}

		for(Instance instance : newInsts)
		{
			int classVal = (int) instance.classValue();
			classCnt2[classVal]++;
		}
		log.print(classCnt2);
		return newInsts;
	}
	
	/**
	 * Remove total ratio"s values
	 * @param instances
	 * @return
	 * @throws IllegalEnumValue
	 * @throws DataFileReadError
	 * @throws DataFileNotFind
	 * @throws DataSourceLoadError
	 * @throws TimerException
	 * @throws IOException
	 */
	public static Instances removeValues(Instances instances) throws IllegalEnumValue, DataFileReadError, DataFileNotFind, DataSourceLoadError, TimerException, IOException
	{
		int TOTAL_VALUES = 0;
		for(Instance instance : instances)
		{
			for(int i = 0; i < instance.numAttributes() - 1; i++)
			{
				if (instance.value(i) > 0)
				{
					TOTAL_VALUES++;
				}
			}
		}
		double leftValues = (1 - SPARSE_RATE) * TOTAL_VALUES;
		log.print("leftValues= " + leftValues + ", sparse_rate=" + SPARSE_RATE + ", total_values = " + TOTAL_VALUES);
		int MIN_ATTR = (int) (leftValues / instances.size() - 0.499999999999);
		log.print("MIN_ATTR=" + MIN_ATTR);
		Instances newInstances = new Instances(instances);
		Random random = new Random(1);
		int valueCnt = 0, changeCnt = 0;
		for(Instance instance : newInstances)
		{
			int tmpCnt = 0, tmpChgCnt = 0;
			List<Integer> lstAttrs = new ArrayList<Integer>();
			for(int i = 0; i < instance.numAttributes() - 1; i++)
			{
				if (instance.value(i) > 0)
				{
					lstAttrs.add(i);
					valueCnt ++; tmpCnt ++;
				}
			}
			
			int leftAttrCnt = lstAttrs.size();
			double filterRatio = (double)MIN_ATTR / lstAttrs.size();
			for(int index : lstAttrs)
			{
				if (leftAttrCnt <= MIN_ATTR)
				{
					break;
				}
				if (random.nextDouble() > filterRatio)
				{
					changeCnt++; tmpChgCnt++;
					instance.setValue(index, 0);
					leftAttrCnt--;
				}
			}
			
			int dist = tmpCnt - tmpChgCnt;
//			NormalizeArray(instance);
			
			
//			log.printNoAddon(dist + "," + tmpCnt + "," + tmpChgCnt);
		}
		
		log.print("Total " + valueCnt + " values and " + changeCnt + " values are changed, changed ratio is " + (double)changeCnt/valueCnt);
		return newInstances;
	}
	
	/**
	 * Select 10% or 20% from instances
	 * @param instances
	 * @param ratio
	 * @return
	 */
	public static Instances filterRatio(Instances instances, double ratio)
	{
		Instances insts = new Instances(instances, 0);
		
		instances.randomize(new Random(1));
		int cnt = (int) (instances.size() * ratio);
		for(int i = 0; i < cnt; i++)
		{
			insts.add(instances.get(i));
		}
		
		return insts;
	}
	

	/**
	 * Normalized the array.
	 * @param ar
	 */
	public static void NormalizeArray(Instance instance)
	{
		int iSum = 0;
		for(int i = 0; i < instance.numAttributes() - 1; i++)
		{
			iSum += instance.value(i);
		}
		for(int i = 0; i < instance.numAttributes() - 1; i++)
		{
			instance.setValue(i, instance.value(i)/iSum);
		}
	}
}
