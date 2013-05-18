package dataHandling;
//package Framework;
//
//import java.util.Random;
//
//import weka.core.Instance;
//import weka.core.Instances;
//import Share.Comm;
//import Share.Option;
//import Share.log;
//
///**
// * Put the majority instances at the head [0..majorEnd], and put the minority instances at the rear [majorEnd+1..N].
// * @author user
// *
// */
//public class DataInsts extends Instances{
//
//	/**
//	 * The start position for minority class. 
//	 * {majority...,   |  minority...}
//	 *                /|\
//	 *                 |
//	 *             minorStart 
//	 */
//	public int minorStart = 0;
//	
//	private static final long serialVersionUID = 1L;
//
//	public DataInsts(Instances dataset) throws Exception
//	{
//		super(dataset, 0);
//		init(dataset);
//	}
//
//	public DataInsts(Instances dataset, int capacity)
//	{
//		super(dataset, capacity);
//		this.minorStart = 0;
//	}
//	
//	public void init(Instances instances) throws Exception
//	{		
//		// get all class labels
//		int numClass = instances.numClasses();
//		if (numClass != 2)
//		{
//			log.print("The number of class (" + numClass + ") is not equal to 2!");
//			throw new Exception();
//		}
//		
//		// split it into minority and majority class.
//		Instances MinorInsts = new Instances(instances, 0);
//		Instances MajorInsts = new Instances(instances, 0);
//		for(Instance instance : instances)
//		{
//			double cls = instance.classValue();
//			if (cls == Comm.MINOR_CLASS)
//			{
//				MinorInsts.add(instance);
//			}
//			else
//			{
//				MajorInsts.add(instance);
//			}
//		}
//		
//		this.minorStart = MajorInsts.size();
//		this.m_Instances.clear();
//		MajorInsts.randomize(new Random(1));
//		MinorInsts.randomize(new Random(1));
//		this.m_Instances.addAll(MajorInsts);
//		this.m_Instances.addAll(MinorInsts);	
//	}
//	
//	public Instances getMajorInsts() throws Exception
//	{
//		return InstancesHelper.drawSubList(this, 0, this.minorStart - 1);
//	}	
//	
//	public void addMajorInsts(Instances instances)
//	{
//		this.m_Instances.addAll(0, instances);
//		this.minorStart += instances.size();
//	}
//	
//	public Instances getMinorInsts() throws Exception
//	{
//		if (this.minorStart > 0)
//		{
//			return InstancesHelper.drawSubList(this, this.minorStart, this.size() - 1);
//		}
//		else
//		{
//			log.print("ERROR: minor start is zero!!!");
//			return null;
//		}
//	}
//
//	public void addMinorInsts(Instances instances)
//	{
//		this.m_Instances.addAll(instances);
//	}
//	
//}
