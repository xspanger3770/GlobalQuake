package globalquake.ui.settings;

import javax.swing.JPanel;
import java.text.NumberFormat;
import java.text.ParseException;
import java.text.ParsePosition;
import java.util.Locale;

public abstract class SettingsPanel extends JPanel{

	public abstract void save() throws NumberFormatException, ParseException;
	
	public abstract String getTitle();

	public void refreshUI() {}

	private final NumberFormat format = NumberFormat.getNumberInstance(Locale.getDefault());

	public Number parse(String str) throws ParseException{
		ParsePosition parsePosition = new ParsePosition(0);
		Number number = format.parse(str, parsePosition);

		if(parsePosition.getIndex() != str.length()){
			throw new ParseException("Invalid input", parsePosition.getIndex());
		}

		return number;
	}

}
