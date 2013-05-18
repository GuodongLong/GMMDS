package Share;

import java.util.Date;

import Exception.TimerException;

public class Timer2 {

	/**
	 * Timer
	 */
	private static int globleId = 0;
	private int  id = 0;
	private Date start = null;
	private Date end = null;
	private String message = "";
	
	public Timer2()
	{
		id = getNewId();
		this.message = "";
	}

	public Timer2(String msg)
	{
		id = getNewId();
		this.message = msg;
	}
	
	public int getNewId()
	{
		return globleId++;
	}
	
	@SuppressWarnings("deprecation")
	public Date startTimer()
	{
		start = new Date(System.currentTimeMillis()); 
		end = new Date(System.currentTimeMillis()); ;
		log.print("*************************************Begin (id=" + id + ")" + this.message + ":" + start.toLocaleString() + "*************************************\n");
		return start;
	}
	
	@SuppressWarnings("deprecation")
	public Date stopTimer()
	{
		end = new Date(System.currentTimeMillis());
		log.print("*************************************End: (id=" + id + ")" + this.message + ":" + end.toLocaleString() + "*************************************\n");
		
		/* Print the duration of timer */
		getDuration();
		return end;
	}
	
	/**
	 * Print the duration of timer. It is called by stopTimer function.
	 * @return
	 */
	private long getDuration()
	{
		long dur = end.getTime() - start.getTime();
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

