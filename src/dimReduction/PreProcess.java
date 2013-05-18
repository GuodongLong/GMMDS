package dimReduction;

import weka.core.Instance;
import weka.core.Instances;

public class PreProcess {

	public Instances decentralize(Instances instances)
	{
		Instances newInsts = new Instances(instances);
		
		for(int attrid = 0; attrid < instances.numAttributes() - 1; attrid++)
		{
			double avgValue = 0;
			for(Instance instance : instances)
			{
				avgValue += instance.value(attrid);
			}
			avgValue /= instances.numAttributes() - 1;

			for(Instance instance : instances)
			{
				double newValue = instance.value(attrid) - avgValue;
				instance.setValue(attrid, newValue);
			}
		}
		
		return newInsts;
	}
}
