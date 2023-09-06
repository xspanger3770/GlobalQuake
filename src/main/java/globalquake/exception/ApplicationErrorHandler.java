package globalquake.exception;

import globalquake.exception.action.CloseAction;
import globalquake.exception.action.IgnoreAction;
import globalquake.exception.action.TerminateAction;
import org.tinylog.Logger;

import javax.swing.*;
import java.awt.*;
import java.io.PrintWriter;
import java.io.StringWriter;

public class ApplicationErrorHandler implements Thread.UncaughtExceptionHandler {

	private Window parent;

	private int errorCount = 0;


	public ApplicationErrorHandler(Window parent) {
		this.parent = parent;
	}

	public void setParent(Window parent) {
		this.parent = parent;
	}

	@Override
	public void uncaughtException(Thread t, Throwable e) {
		Logger.error("An uncaught exception has occurred in thread {} : {}", t.getName(), e.getMessage());
		Logger.error(e);
		handleException(e);
	}

	public synchronized void handleException(Throwable e) {
		if (!(e instanceof RuntimeApplicationException)) {
			showDetailedError(e);
			return;
		}

		if (e instanceof FatalError ex) {
			showGeneralError(ex.getUserMessage(), true);
		} else {
			ApplicationException ex = (ApplicationException) e;
			showGeneralError(ex.getUserMessage(), false);
		}
	}

	public synchronized void handleWarning(Throwable e) {
		showWarning(e.getMessage());
	}

	private void showWarning(String message) {
		JOptionPane.showMessageDialog(parent, message, "Warning", JOptionPane.WARNING_MESSAGE);
	}

	private void showDetailedError(Throwable e) {
		errorCount++;
		if (errorCount == 2) {
			System.exit(0);
		}
		final Object[] options = getOptionsForDialog(true);
		JOptionPane.showOptionDialog(parent, createDetailedPane(e), "Fatal Error", JOptionPane.DEFAULT_OPTION,
				JOptionPane.ERROR_MESSAGE, null, options, null);
		errorCount = 0;
	}

	private Component createDetailedPane(Throwable e) {
		JTextArea textArea = new JTextArea(16, 60);
		textArea.setEditable(false);
		StringWriter stackTraceWriter = new StringWriter();
		e.printStackTrace(new PrintWriter(stackTraceWriter));
		textArea.append(stackTraceWriter.toString());
		return new JScrollPane(textArea);
	}

	private void showGeneralError(String message, boolean isFatal) {
		final String title = isFatal ? "Fatal Error" : "Application Error";
		final Object[] options = getOptionsForDialog(isFatal);

		JOptionPane.showOptionDialog(parent, message, title, JOptionPane.DEFAULT_OPTION, JOptionPane.ERROR_MESSAGE, null,
				options, null);
	}

	private Component[] getOptionsForDialog(boolean isFatal) {
		if (!isFatal) {
			return null; // use default
		}

		return new Component[] { new JButton(new TerminateAction()), new JButton(new CloseAction(parent)),
				new JButton(new IgnoreAction()) };
	}
}
