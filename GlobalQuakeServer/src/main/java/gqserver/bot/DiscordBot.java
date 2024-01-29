package gqserver.bot;

import globalquake.core.GlobalQuake;
import globalquake.core.earthquake.data.Earthquake;
import globalquake.core.events.GlobalQuakeEventListener;
import globalquake.core.events.specific.QuakeCreateEvent;
import globalquake.core.events.specific.QuakeRemoveEvent;
import globalquake.core.events.specific.QuakeReportEvent;
import globalquake.core.events.specific.QuakeUpdateEvent;
import net.dv8tion.jda.api.EmbedBuilder;
import net.dv8tion.jda.api.JDA;
import net.dv8tion.jda.api.JDABuilder;
import net.dv8tion.jda.api.events.session.ReadyEvent;
import net.dv8tion.jda.api.hooks.ListenerAdapter;
import net.dv8tion.jda.api.requests.GatewayIntent;

import javax.security.auth.login.LoginException;

import globalquake.core.Settings;

public class DiscordBot extends ListenerAdapter{

    private static JDA jda;

    public static void init() throws LoginException {
        jda = JDABuilder.createDefault(Settings.discordBotToken).enableIntents(GatewayIntent.GUILD_MESSAGES)
                .addEventListeners(new DiscordBot())
                .build();

        GlobalQuake.instance.getEventHandler().registerEventListener(new GlobalQuakeEventListener() {
            @Override
            public void onQuakeCreate(QuakeCreateEvent event) {
                sendQuakeCreateInfo(event.earthquake());
            }

            @Override
            public void onQuakeUpdate(QuakeUpdateEvent event) {
                sendQuakeUpdateInfo(event.earthquake());
            }

            @Override
            public void onQuakeRemove(QuakeRemoveEvent event) {
                sendQuakeRemoveInfo(event.earthquake());
            }

            @Override
            public void onQuakeReport(QuakeReportEvent event) {
                sendQuakeReportInfo(event);
            }
        });
    }

    private static void sendQuakeReportInfo(QuakeReportEvent event) {
        if(jda == null){
            return;
        }
    }

    private static void sendQuakeRemoveInfo(Earthquake earthquake) {
        if(jda == null){
            return;
        }
    }

    private static void sendQuakeUpdateInfo(Earthquake earthquake) {
        if(jda == null){
            return;
        }

    }

    private static void sendQuakeCreateInfo(Earthquake earthquake) {
        if(jda == null){
            return;
        }

        var channel = jda.getGuildById(Settings.discordBotGuildID).getTextChannelById(Settings.discordBotChannelID);
        EmbedBuilder builder = new EmbedBuilder();
        builder.setTitle("M%.1f %s".formatted(earthquake.getMag(), earthquake.getRegion()));
        builder.setDescription("Description of the Embed");
        builder.appendDescription("asdasda");

        channel.sendMessageEmbeds(builder.build()).queue();
    }

    @Override
    public void onReady(ReadyEvent event) {
        System.out.println("Bot is ready!");
        var channel = jda.getGuildById(Settings.discordBotGuildID).getTextChannelById(Settings.discordBotChannelID);
        EmbedBuilder builder = new EmbedBuilder();
        builder.setTitle("title");
        builder.setDescription("Description of the Embed");
        builder.appendDescription("asdasda");

        channel.sendMessageEmbeds(builder.build()).queue();
    }

}
