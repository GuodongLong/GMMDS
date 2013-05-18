package Share;

import java.util.Date;

import Exception.TimerException;

public class Timer {

	/**
	 * Timer
	 */
	private static Date start = null;
	private static Date end = null;
	private static boolean isUsing = false;

	@SuppressWarnings("deprecation")
	public static Date startTimer() throws TimerException
	{
		if (isUsing)
		{
			log.print("The timer is busy now. Please try it later.");
			throw new TimerException();
		}
		isUsing = true;
		start = new Date(System.currentTimeMillis()); 
		end = null;
		log.print("*************************************Begin:" + start.toLocaleString() + "*************************************\n");
		return start;
	}
	
	@SuppressWarnings("deprecation")
	public static Date stopTimer() throws TimerException
	{
		if (!isUsing)
		{
			log.print("The timer is not start. Please start it.");
			throw new TimerException();
		}
		isUsing = false;
		end = new Date(System.currentTimeMillis());
		log.print("*************************************End:" + end.toLocaleString() + "*************************************\n");
		
		/* Print the duration of timer */
		getDuration();
		return end;
	}
	
	/**
	 * Print the duration of timer. It is called by stopTimer function.
	 * @return
	 */
	private static long getDuration()
	{
		long dur = end.getTime() - start.getTime();
		start = null;
		end   = null;
		log.print("timer duration " + dur + " millseconds = " + dur/1000 + " seconds.");
		return dur;
	}
	
	public static void main(String[] args) throws TimerException, InterruptedException
	{
		Timer.startTimer();
		Thread.sleep(10000);
		Timer.stopTimer();
	}
}
