package dataHandling;

import weka.core.Instance;
import weka.core.Instances;


public class Normalization {
	
	/**
	 * Normalize one given instances.
	 * @param instances
	 */
	public static void Normalize(Instances instances)
	{
		double sumWeight = 0;
		
		for(Instance sample: instances)
		{
			sumWeight += sample.weight();
		}
		
		for(Instance sample: instances)
		{
			sample.setWeight(sample.weight() / sumWeight);
		}
	}
	
	/**
	 * Normalize the given array.
	 * @param distribution
	 */
	public static void Normalize(double[] distribution)
	{
		double sumVal = 0;
		for(int i = 0; i < distribution.length; i++)
		{
			sumVal += distribution[i];
		}

		for(int i = 0; i < distribution.length; i++)
		{
			distribution[i] = distribution[i] / sumVal;
		}
	}
}
