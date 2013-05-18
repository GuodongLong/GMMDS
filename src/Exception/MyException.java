package Exception;


public class MyException extends Exception{
	
	/**
	 * @uml.property  name="errorMessage"
	 */
	String errorMessage;
	/**
	 * @uml.property  name="errorCode"
	 */
	String errorCode;
	
	public MyException(){
		
	}
	
	public MyException(String message){
		super(message);
		this.setErrorMessage(message);
	}

	/**
	 * @author
	 * @date  22/09/2011
	 * @precondition
	 * @return
	 * @uml.property  name="errorMessage"
	 */
	public String getErrorMessage() {
		return errorMessage;
	}

	/**
	 * @author
	 * @date  22/09/2011
	 * @precondition
	 * @param errorMessage
	 * @uml.property  name="errorMessage"
	 */
	public void setErrorMessage(String errorMessage) {
		this.errorMessage = errorMessage;
	}

	/**
	 * @author
	 * @date  22/09/2011
	 * @precondition
	 * @return
	 * @uml.property  name="errorCode"
	 */
	public String getErrorCode() {
		return errorCode;
	}

	/**
	 * @author
	 * @date  22/09/2011
	 * @precondition
	 * @param errorCode
	 * @uml.property  name="errorCode"
	 */
	public void setErrorCode(String errorCode) {
		this.errorCode = errorCode;
	}
	
}