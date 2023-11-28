package globalquake.ui.action;

import org.tinylog.Logger;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;

public class OpenURLAction extends AbstractAction {

    private final String url;

    public OpenURLAction(String url, String name) {
        super(name);
        this.url = url;
    }

    public void openWebpage(URI uri) {
        Desktop desktop = Desktop.isDesktopSupported() ? Desktop.getDesktop() : null;
        if (desktop != null && desktop.isSupported(Desktop.Action.BROWSE)) {
            try {
                desktop.browse(uri);
            } catch (Exception e) {
                Logger.error(e);
            }
        }
    }

    public void openWebpage() {
        try {
            openWebpage(new URL(url).toURI());
        } catch (URISyntaxException | MalformedURLException e) {
            Logger.error(e);
        }
    }

    @Override
    public void actionPerformed(ActionEvent actionEvent) {
        openWebpage();
    }
}
