package dataHandling;

import java.io.File;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import Exception.DataFileNotFind;
import Exception.DataFileReadError;
import Exception.DataSourceLoadError;
import Exception.IllegalEnumValue;
import Exception.TimerException;
import Share.FileOption;
import Share.log;

public class DataLoader {

	/**
	 * Load data from file.
	 * @param fileName
	 * @throws DataFileReadError 
	 * @throws DataFileNotFind 
	 * @throws DataSourceLoadError 
	 * @throws TimerException 
	 * @throws Exception, IllegalEnumValue 
	 */
	public static Instances loadDataFromFile(String fileName) throws IllegalEnumValue, DataFileReadError, DataFileNotFind, DataSourceLoadError, TimerException
	{
		log.print("Begin load data from file " + fileName);
		
		/* Validate file */
		File file = new File(fileName);
		if (!file.exists() || !file.isFile())
		{
			log.print("File " + fileName + " is not exist or is not a real file.");
			throw new DataFileNotFind();
		}
		
		/* Get instances */
		Instances instances = null;
		try {
			instances = DataSource.read(fileName);//dataSource.getDataSet(0);
			instances.setClassIndex(instances.numAttributes() - 1);
		} catch (Exception e) {
			log.print("The " + fileName + " cannot be loaded correctly." + e.toString());
			e.printStackTrace();
			throw new DataSourceLoadError();
		}
		
		log.print("Total " + instances.size() + " samples were loaded!");
		
		return instances;
	}
	
	/**
	 * Load data for the 20NG_CS_in_domain
	 * @return
	 * @throws IllegalEnumValue
	 * @throws DataFileReadError
	 * @throws DataFileNotFind
	 * @throws DataSourceLoadError
	 * @throws TimerException
	 */
	public static Instances loadCSinDomain() throws IllegalEnumValue, DataFileReadError, DataFileNotFind, DataSourceLoadError, TimerException
	{
		String fileName = FileOption.CS_IN_DOMAIN_Nominal;
		Instances instances = loadDataFromFile(fileName); // 0-minority, 1-majority

		Random random = new Random(1);
		for(Instance instance : instances)
		{
			if ((int)instance.classValue() == 0 && random.nextFloat() > 0.02)
			{
				instances.remove(instance);
				if(instances.size() == 100)
				{
					break;
				}
			}
		}

		log.print("Total " + instances.size() + " samples will be used!");
		return instances;
	}
}
