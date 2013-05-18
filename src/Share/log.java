package Share;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Date;
import java.util.List;


/**
 * The common shared functions of the project.
 * @author Gordon
 * @version 04/10/2011 Create
 * @copyright Gordon
 */
public class log {

	/**
	 * The path of log file.
	 * @see Share
	 */
	
	private static String logFile = Comm.logPath + "log.txt";
	
	/**
	 * Print the message into command station and log file.
	 * if print with hasAddon, the message will include the time and caller information. Otherwise only print the original information.
	 * @param str
	 * @param hasAddon
	 * @see Share
	 */
	public static void logBase(String msg, boolean hasAddon)
	{
		if (hasAddon)
		{
			Date dt = new Date(System.currentTimeMillis());
			java.text.SimpleDateFormat df = new java.text.SimpleDateFormat("yyyy/MM/dd_HH:mm:ss ");
			String newMsg = transferMsg(msg);
			newMsg = df.format(dt) + newMsg;
			System.out.println(newMsg);
			writeLog(newMsg);
		}
		else
		{
			System.out.println(msg);
			writeLog(msg);
		}
	}
	
	public static void print(String msg)
	{
		logBase(msg, true);
	}
	

	public static void printNoAddon(String msg)
	{
		logBase(msg, false);
	}
	
	public static void print(String[] msg)
	{
		StringBuilder sb = new StringBuilder();
		for(String str: msg)
		{
			sb.append(str);
			sb.append(" ");
		}
		logBase(sb.toString(), true);
	}
	
	public static void print(double[] msg)
	{
		StringBuilder sb = new StringBuilder();
		for(double d: msg)
		{
			sb.append(d);
			sb.append(" ");
		}
		logBase(sb.toString(), true);
	}
	
	public static void print(double[] msg, int size)
	{
		StringBuilder sb = new StringBuilder();
		int idx = 0;
		for(double d: msg)
		{
			if (idx++ >= size)
			{
				break;
			}
			sb.append(d);
			sb.append(" ");
		}
		logBase(sb.toString(), true);
	}

	public static void print(int[] msg)
	{
		StringBuilder sb = new StringBuilder();
		for(double d: msg)
		{
			sb.append(d);
			sb.append(" ");
		}
		logBase(sb.toString(), true);
	}
	
	public static void print(double[][] data)
	{
		for(int i = 0; i < data.length; i++)
		{
			print(data[i]);
		}
	}
	public static void print(int[][] data)
	{
		for(int i = 0; i < data.length; i++)
		{
			print(data[i]);
		}
	}
	
	public static void print(int i)
	{
		logBase(String.valueOf(i), true);
	}
	
	public static void print(double d)
	{
		logBase(String.valueOf(d), true);
	}
	
	public static void print(Object o)
	{
		logBase(String.valueOf(o), true);
	}
	
	public static void print(List<Object> list)
	{
		list.iterator();
		StringBuilder sb = new StringBuilder();
		for(Object d: list.toArray())
		{
			sb.append(d);
			sb.append(" ");
		}
		logBase(sb.toString(), true);
	}
	/**
	 * Transfer the msg with the new format "className + funcName + msg".
	 * This function only be called by the inner print().
	 */
	private static String transferMsg(String msg)
	{
		String className = null;
		String funcName  = null;
		
		boolean doNext   = false;
		StackTraceElement e[] = Thread.currentThread().getStackTrace();
		/* *****************************************
		 * The elements in e[]:
		 *   -getStackTrace()
		 *   -transferMsg()
		 *   -logBase()
		 *   -print()
		 *   -caller
		 *   ...
		 * ***************************************** */
		for(StackTraceElement s: e)
		{
			if (doNext)
			{
				className = s.getClassName();
				funcName = s.getMethodName();
				break;
			}
			doNext = s.getMethodName().equals("print");
		}
		
		String str = "" + className + "." + funcName + "():" + msg;
		return str;
	}
	
	/**
	 * Record the message into log.
	 * @param str
	 * @see Share
	 */
	public static void writeLog(String str)
	{
		File fout = new File(logFile);

		FileWriter fos = null;
		try {
			if (!fout.exists())
			{
				fout.createNewFile();
			}
			fos = new FileWriter(fout, true);
			fos.append(str);
			fos.append("\n");
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		finally
		{
			if (fos != null)
			{
				try {
					fos.close();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}
		}
	}

	
	
	
}
