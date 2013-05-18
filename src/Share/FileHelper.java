package Share;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;


public class FileHelper {

	//"her", "my", "she", "he", "his", "A", "with", "that", "who", "which", "is", "are", "for", "of",  "to", "in", "it", "the", 
	public static final String[] ENGLISH_STOP_WORDS = {
		"don't", "can't", 
		"a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"};



	
	/**
	 * Clean directories.
	 */
	public static void cleandirectory(String dir)
	{
	    File file = new File(dir);
	    if (!file.exists())
	    {
	    	return;
	    }
	    if (!file.isDirectory()) 
	    {
	    	return;
	    }
	      
        String[] srfilelist = file.list();
        for (int i = 0; i < srfilelist.length; i++) 
        {
	        File dtFile = new File(dir + srfilelist[i]);
	        dtFile.delete();
        }	
	}
	
	/**
	 * Copy one directories' files to another directory.
	 * Cannot copy the sub-directories.
	 * @param srDir
	 * @param dtDir
	 */
	public static void copydirectory(String srDir, String dtDir)
	{
	      File file = new File(srDir);
	      if (file.isDirectory()) 
	      {	
	          String[] srfilelist = file.list();
	          for (int i = 0; i < srfilelist.length; i++) 
	          {
	        	  String dtfilepath = dtDir + srfilelist[i]; 
	        	  File dtFile = new File(dtfilepath);
	        	  if (!dtFile.exists())
		  		  {
	        		  try {
						  dtFile.createNewFile();
					  } catch (IOException e) {
						  e.printStackTrace();
					  }
		  		  }
	        	  copyfile(srDir + srfilelist[i], dtfilepath);
	          }
	      }
	}
	
	/**
	 * Copy one file from source path to destination path.
	 * @param srFile
	 * @param dtFile
	 */
	public static void copyfile(String srFile, String dtFile)
	{
		try
		{
			if (srFile == null)
			{
			    FileWriter fout = new FileWriter(dtFile);
			    fout.write("");
			    fout.close();
			    return;
			}
			
		    File f1 = new File(srFile);
		    File f2 = new File(dtFile);
		    InputStream in = new FileInputStream(f1);
		  
		    //For Append the file.
		    //OutputStream out = new FileOutputStream(f2,true);

		    //For Overwrite the file.
		    OutputStream out = new FileOutputStream(f2);

		    byte[] buf = new byte[1024];
		    int len;
		    while ((len = in.read(buf)) > 0)
		    {
		    	out.write(buf, 0, len);
		    }
		    
			in.close();
			out.close();
			System.out.println("File copied.");
		}
		catch(FileNotFoundException ex){
			System.out.println(ex.getMessage() + " in the specified directory.");
			System.exit(0);
		}
		catch(IOException e){
			System.out.println(e.getMessage());  
		}
	}
	
	public static void appendFile(String srFile, String dtFile)
	{
		try
		{
		    File f1 = new File(srFile);
		    File f2 = new File(dtFile);
		    InputStream in = new FileInputStream(f1);
		  
		    //For Append the file.
		    OutputStream out = new FileOutputStream(f2,true);

		    //For Overwrite the file.
//		    OutputStream out = new FileOutputStream(f2);

		    int offset = (int) f2.length();
		    byte[] buf = new byte[1024];
		    int len;
		    while ((len = in.read(buf)) > 0)
		    {
		    	out.write(buf, 0, len);
		    	offset += len;
		    }
		    
			in.close();
			out.close();
			System.out.println("File copied. Total " + offset + " bytes.");
		}
		catch(FileNotFoundException ex){
			System.out.println(ex.getMessage() + " in the specified directory.");
			System.exit(0);
		}
		catch(IOException e){
			System.out.println(e.getMessage());  
		}
	}
	
	public static void combineFile(String[] srcFiles, String dstFile)
	{
		if (srcFiles == null || srcFiles.length < 1)
		{
			log.print("source files is null!");
		}
		copyfile(srcFiles[0], dstFile);
		for(int i = 1; i < srcFiles.length; i++)
		{
			appendFile(srcFiles[i], dstFile);
		}
	}
	

	private static Date start = null;
	private static Date end = null;

	public static Date startTimer()
	{
		start = new Date(System.currentTimeMillis()); 
		return start;
	}
	
	public static Date stopTimer()
	{
		end = new Date(System.currentTimeMillis());
		return end;
	}
	
	public static long getDuration()
	{
		long dur = end.getTime() - start.getTime();
		start = null;
		end   = null;
		log.print("timer duration " + dur + " millseconds.");
		return dur;
	}
	
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
	 * Transfor the HashMap to a two dimension array -- ar[2][map.size];
	 * ar[0] store the key array;
	 * ar[1] store the value array;
	 * @param map
	 * @return
	 */
	public static Object[] HashMapToArray(HashMap<Object, Object> hm)
	{
		Object[][] ar = new Object[2][hm.size()];
		int idx = 0;
		Iterator<Entry<Object, Object>> it = hm.entrySet().iterator();
		while(it.hasNext())
		{
			Map.Entry<Object, Object> entry = it.next();
			Object key   = entry.getKey();
			Object value = entry.getValue();
			ar[0][idx] = key;
			ar[1][idx] = value;
			idx++;
		}
		
		return ar;
	}
	

	/**
	 * Write the HashMap into file.
	 * @param output
	 * @throws IOException
	 */
	@SuppressWarnings("rawtypes")
	public static void HashMapToFile( HashMap hm, File output, String delimeter)
	{
		FileWriter fw;
		try {
			fw = new FileWriter(output);
			@SuppressWarnings("unchecked")
			Iterator<Entry<Object, Object>> it = hm.entrySet().iterator();
			while(it.hasNext())
			{
				Map.Entry<Object, Object> entry = it.next();
				fw.write(entry.getKey() + delimeter + entry.getValue() + "\n");
				
				String str = (String)entry.getKey();
				if (str.equalsIgnoreCase("youth"))
				{
					log.print("youth is found.");
				}
			}
			
			fw.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

	public static int getNumLinesInFile(String filePath) throws IOException
	{
		return getNumLinesInFile(filePath, 0);
	}
	
	public static int getNumLinesInFile(String filePath, int BlankLineSize) throws IOException
	{
		File file = new File(filePath);
		return getNumLinesInFile(file, BlankLineSize);
	}
	/**
	 * Get the available line number of the given file.
	 * @param input
	 * @return
	 * @throws IOException
	 */
	public static int getNumLinesInFile(File input) throws IOException
	{
		return getNumLinesInFile(input, 0);
	}
	
	
	/**
	 * Get the available line number of the given file.
	 * @param input
	 * @return
	 * @throws IOException
	 */
	public static int getNumLinesInFile(File input, int BlankLineSize) throws IOException
	{
		FileReader fr = new FileReader(input);
		BufferedReader br = new BufferedReader(fr);
		
		String line = null;
		int iCnt    = 0;
		while((line = br.readLine()) != null)
		{
			if (line.length() <= BlankLineSize)
			{
				continue;
			}
			iCnt++;
		}
		
		br.close();
		fr.close();
		
		return iCnt;
	}
}
