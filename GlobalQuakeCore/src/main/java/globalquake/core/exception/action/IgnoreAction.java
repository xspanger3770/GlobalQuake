package globalquake.core.exception.action;

import java.awt.Component;
import java.awt.Window;
import java.awt.event.ActionEvent;
import java.awt.event.KeyEvent;

import javax.swing.AbstractAction;
import javax.swing.SwingUtilities;

public class IgnoreAction extends AbstractAction {

    public IgnoreAction() {
        super("Ignore");
        putValue(SHORT_DESCRIPTION, "Ignore the error and continue (unsafe)");
        putValue(MNEMONIC_KEY, KeyEvent.VK_I);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        Window w = SwingUtilities.getWindowAncestor((Component) e.getSource());

        if (w != null) {
            w.setVisible(false);
        }
    }

}
